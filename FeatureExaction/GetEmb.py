#!/usr/bin/env python
import json
import torch
import argparse
import logging
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import log_softmax
from tqdm import tqdm
import demjson3
import os
from typing import Generator, List, Dict, Any, Iterable
import sys
import re
from captum.attr import IntegratedGradients
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM inference with hidden state and logprob extraction.")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the local model.")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Input jsonl file path, supports wildcard like 'data/*.jsonl'.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output jsonl file path.")
    parser.add_argument("--target_phrases", type=str, nargs="+", default=[],
                        help="List of phrases to extract embeddings for (only in prompt).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use, e.g., 'cuda:0'. If not set, auto-detect. Multi-GPU via device_map='auto'.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. 0.0 for greedy.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision.")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Use fast tokenizer (recommended).")
    parser.add_argument("--IUIS", action="store_true", help="extract IUIS features.")
    parser.add_argument("--IGscore", action="store_true", help="extract IG scores.")
    return parser.parse_args()

def load_jsonl_files(file_pattern: str) -> Generator[Dict, None, None]:
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files match pattern: {file_pattern}")
    logger.info(f"Found {len(files)} files: {files}")
    for file_path in files:
        logger.info(f"Loading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in {file_path} line {line_num}: {e}")

def write_jsonl_stream(data_iter: Iterable[Dict], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_iter:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

class LLMInference:
    def __init__(self, args):
        self.args = args

        if args.device is None:
            self.device_map = "auto" if torch.cuda.is_available() else None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_map = "auto"
            self.device = args.device

        logger.info(f"Using device_map: {self.device_map}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if args.fp16:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        logger.info(f"Loading model with dtype: {dtype}, device_map: {self.device_map}")

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=self.device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            offload_folder=None,
            attn_implementation="eager"
        )
        self.model.eval()



    def prepare_messages_for_swift_inference(self, messages):
        if not messages:
            return messages

        new_messages = []
        for i, msg in enumerate(messages):
            new_msg = msg.copy()
            if isinstance(msg.get('content'), str):
                if i < len(messages) - 1 :
                    new_msg['content'] = msg['content'].strip()
                else:
                    continue
            new_messages.append(new_msg)
        return new_messages

    def extract_phrase_embeddings(self, input_ids, last_hidden_state, offset_mapping, prompt, messages, target_phrases,
                                  tokenizer, attentions=None, token_attributions=None, layer_id=-1):
        """
            {
                "phrase": {
                    "mean_embedding": [...],
                    "token_embeddings": [[...], [...]],
                    "token_indices": [5, 6],
                    "char_span": [20, 26],
                    "matched_tokens_count": 2
                }
            }
        """
        token_offsets = offset_mapping[0].cpu().tolist()  # (seq_len, 2)

        valid_tokens = []
        for idx, (start, end) in enumerate(token_offsets):
            if start == 0 and end == 0:
                continue
            if 0 <= start < end <= len(prompt):
                valid_tokens.append((idx, start, end))
        # print(f"Valid tokens: {len(valid_tokens)}")

        if attentions is not None:
            # layer_id = len(attentions)//2
            last_layer_attn = attentions[layer_id]  # (1, num_heads, out_seq, in_seq)
            # print(f"Attention dtype: {last_layer_attn.dtype}")
            # print(f"Attention shape: {last_layer_attn.shape}")
            # print(f"Output seq length: {last_layer_attn.shape[2]}")
            # print(f"prompt length: {input_ids.shape[1]}")


            head_avg_attn = last_layer_attn.mean(dim=1)  # (1, out_seq, in_seq)
            in_attn_all = head_avg_attn[0, input_ids.shape[1], :]  # (in_seq,)
            # print(f"attn length before: {in_attn_all.shape[0]}")
            in_attn = in_attn_all[:input_ids.shape[1]]
            # print(f"attn length after: {in_attn.shape[0]}")
        else:
            print("No attentions provided.")
            in_attn = torch.ones(input_ids.shape[1], device=input_ids.device, dtype=torch.float32) / input_ids.shape[1]

        embeddings = {}
        for i, phrase in enumerate(target_phrases):
            is_last = (i == len(target_phrases) - 1)
            if is_last:
                # matched_token_indices = -1
                last_token_idx, last_start, last_end = valid_tokens[-1]
                matched_token_indices = [last_token_idx]
                phrase_start = last_start
                phrase_end = last_end
            else:
                phrase_start = prompt.rfind(phrase)
                if phrase_start == -1:
                    phrase_striped = phrase.strip()
                    phrase_start = prompt.rfind(phrase_striped)
                    if phrase_start == -1:
                        logger.warning(f"Phrase '{phrase}' not found in prompt.")
                        embeddings[phrase] = None
                        continue

                phrase_end = phrase_start + len(phrase)

                matched_token_indices = []
                for token_idx, tok_start, tok_end in valid_tokens:
                    if not (tok_start >= phrase_end or tok_end <= phrase_start):
                        matched_token_indices.append(token_idx)

                if not matched_token_indices:
                    logger.warning(f"Phrase '{phrase}' found in text but no matching tokens.")
                    embeddings[phrase] = None
                    continue

            phrase_hs = last_hidden_state[0, matched_token_indices, :]  # (k, H)
            phrase_hs = torch.mean(phrase_hs, dim=0)  # (H,) 
            phrase_hs = phrase_hs.float().cpu().numpy()
            # phrase_emb_mean = torch.mean(phrase_hs, dim=0).cpu().numpy()

            att_weights = in_attn[matched_token_indices].mean()  # (k,)
            att_weights = att_weights.cpu().numpy()

            ig_score = None
            if token_attributions is not None:
                ig_vals = token_attributions[0, matched_token_indices]  # [k]
                ig_score = ig_vals.mean().item()

            matched_ids = input_ids[0, matched_token_indices]
            tokens = tokenizer.convert_ids_to_tokens(matched_ids)

            embeddings[phrase] = {
                "mean_embedding": phrase_hs.tolist(),
                "phrase_importance": att_weights.tolist(),
                "ig_attribution": ig_score,
                # "token_embeddings": phrase_emb_tokens.tolist(),
                "token_indices": matched_token_indices,
                "token_phrases": tokens,
                "char_span": [phrase_start, phrase_end],
                "matched_tokens_count": len(matched_token_indices)
            }

        return embeddings

    def get_logprobs_from_scores(self, scores: tuple, generated_ids: torch.Tensor, input_ids: torch.Tensor):
        batch_size = generated_ids.shape[0]
        prompt_len = input_ids.shape[1]
        all_logprobs = []

        # decoded_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print(f"Generated text: {decoded_output}")

        for i in range(batch_size):
            gen_ids = generated_ids[i, prompt_len:]
            min_len = min(len(scores), len(gen_ids))
            logprobs = []

            for step in range(min_len):
                logits = scores[step][i].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                token_id = gen_ids[step].item()
                logp = log_probs[token_id].item()
                token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                logprobs.append({
                    "token_id": token_id,
                    "token": token_str,
                    "logprob": logp
                })
            all_logprobs.append(logprobs)

        return all_logprobs

    def process_batch(self, messages_list: List[List[Dict]]) -> List[Dict]:
        prompts = [
            self.tokenizer.apply_chat_template(self.prepare_messages_for_swift_inference(msgs), tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]

        data0 = [demjson3.decode(msgs[-1]["content"]) for msgs in messages_list]

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_offsets_mapping=True
        )
        # print(f"Original prompt token length: {inputs['input_ids'].shape[1]}")
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        offset_mapping = inputs["offset_mapping"]  # CPU tensor

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1]  # (B, seq_len, H)
            # last_hidden_state = outputs.last_hidden_state  # (B, seq_len, H)
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": self.args.max_new_tokens,
                "do_sample": self.args.do_sample,
                "temperature": self.args.temperature,
                "pad_token_id": self.tokenizer.pad_token_id,
                "use_cache": True,
                "output_scores": True,
                "return_dict_in_generate": True,
            }
            gen_outputs = self.model.generate(**generate_kwargs)
            generated_ids = gen_outputs.sequences

        if self.args.IGscore:
            embedding_layer = self.model.get_input_embeddings()
            def forward_func(input_embeddings):
                logits = self.model(inputs_embeds=input_embeddings,
                                    attention_mask=attention_mask).logits
                last_token_logits = logits[:, -1, :]
                try:
                    pos_id = self.tokenizer.encode("positive", add_special_tokens=False)[0]
                except:
                    pos_id = self.tokenizer.encode("Positive", add_special_tokens=False)[0]
                return last_token_logits[:, pos_id]

            ig = IntegratedGradients(forward_func)
            input_embeddings = embedding_layer(input_ids)  # [B, seq_len, H]
            baseline = torch.zeros_like(input_embeddings)

            attributions, _ = ig.attribute(
                inputs=input_embeddings,
                baselines=baseline,
                target=None,
                n_steps=50,
                return_convergence_delta=True
            )
            token_attributions = attributions.norm(dim=-1)  # [B, seq_len]
        else:
            token_attributions = None

        full_input_ids = generated_ids  # prompt + output token

        with torch.no_grad():
            full_outputs = self.model(
                input_ids=full_input_ids,
                output_attentions=True,
                attention_mask=(generated_ids != self.tokenizer.pad_token_id).to(generated_ids.device)
            )
        attentions_fp32 = tuple(attn.float().detach() for attn in full_outputs.attentions)

        batch_results = []
        logprobs_list = self.get_logprobs_from_scores(gen_outputs.scores, generated_ids, input_ids)
        for i in range(len(messages_list)):
            messages = messages_list[i]
            data1 = data0[i]
            cpra = data1["cpra"]

            phrase_map = {
            "Clinvar database": "Clinvar database",
            "HGMD database": "HGMD database",
            "ACMG evidence items": "ACMG evidence items",
            "pathogenicity assessment": "Database assessment",
            "functional consequences": "Functional consequences",
            "SIFT": "SIFT",
            "Polyphen2-HDIV": "Polyphen2-HDIV",
            "Polyphen2-HVAR": "Polyphen2-HVAR",
            "MutationTaster2": "MutationTaster2",
            "Revel": "Revel",
            "Condel": "Condel",
            "SpliceAI": "SpliceAI",
            "dbscSnv_RF": "dbscSnv_RF",
            "dbscSnv_ADA": "dbscSnv_ADA",
            "PhyloP Vertebrates": "PhyloP Vertebrates",
            "PhyloP Placental Mammals": "PhyloP Placental Mammals",
            "GERP++": "GERP++",
            "zygosity": "Zygosity",
            "inheritance": "Inheritance mode",
            "1000G": "1000G",
            "GnomAD": "GnomAD",
            "GnomAD-EAS": "GnomAD-EAS",
            "associated diseases": "Disease JSON"
            }


            anno = demjson3.decode(data1["anno"])
            target_phrases = []
            for phrase in self.args.target_phrases:
                try:
                    # tmp = f"\"{phrase}\": \"{anno[phrase]}\""
                    key_str = phrase_map[phrase]
                    val_str = anno[phrase]
                    # if phrase == "associated diseases":
                    #     val_str = json.dumps(val_str)
                    #     val_str = val_str.replace("\\\"", "\"")
                    #     val_str = val_str.replace("\"\"", "\"")
                    #     val_str = val_str.replace("'", "\\'")
                        #val_str = val_str.replace(""", "\\"")
                    tmp = f"{key_str}: {val_str}"
                    #print("tmp:" + tmp)
                    target_phrases.append(tmp)
                except Exception as e:
                    print(f"Error processing '{phrase}': {e}")

            if self.args.IUIS:
                data2 = messages[-7]["content"]
                pattern = r'If\s+"([^"]+)"\s+is\s+on\s+the\s+IEI/PID\s+gene\s+list:\s+(yes|no)'
                IUIS = re.search(pattern, data2, re.IGNORECASE)
                target_phrases = []
                if IUIS:
                    # gene_name = IUIS.group(1)  # "AK2"
                    # is_on_list = IUIS.group(2)  # "yes"
                    target_phrases.append(IUIS.group(0))
                    # print(f"Gene: {gene_name}, On list: {is_on_list}")
                else:
                    print("Pattern not found")

            target_phrases.append(cpra)
            # print(target_phrases)
            full_generated_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=False)
            prompt = prompts[i]
            #print(prompt)
            response = full_generated_text[len(prompt):].strip()

            emb = self.extract_phrase_embeddings(
                input_ids[i].unsqueeze(0),
                last_hidden_state[i].unsqueeze(0),
                offset_mapping[i].unsqueeze(0),
                prompt,
                messages,
                target_phrases,
                self.tokenizer,
                attentions_fp32,
                token_attributions[i].unsqueeze(0)
            )

            batch_results.append({
                #"messages": messages,
                "model_output": {
                    #"prompt": prompt,
                    "response": response,
                    "logprobs": logprobs_list[i],
                    "phrase_embeddings": emb
                }
            })

        return batch_results

def main():
    args = parse_args()
    logger.info(f"Arguments: {args}")

    infer = LLMInference(args)
    data_stream = load_jsonl_files(args.input_jsonl)
    logger.info("Starting inference...")

    def result_generator():
        batch = []
        for item in data_stream:
            batch.append(item)
            if len(batch) == args.batch_size:
                messages_batch = [ex["messages"] for ex in batch]
                try:
                    batch_out = infer.process_batch(messages_batch)
                    for out in batch_out:
                        yield out
                except Exception as e:
                    logger.error(f"Error in batch: {e}")
                    for ex in batch:
                        yield {"messages": ex["messages"], "error": str(e)}
                batch = []

        if batch:
            messages_batch = [ex["messages"] for ex in batch]
            try:
                batch_out = infer.process_batch(messages_batch)
                for out in batch_out:
                    yield out
            except Exception as e:
                logger.error(f"Error in final batch: {e}")
                for ex in batch:
                    yield {"messages": ex["messages"], "error": str(e)}

    write_jsonl_stream(result_generator(), args.output_jsonl)
    logger.info(f"Inference completed. Results saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()
