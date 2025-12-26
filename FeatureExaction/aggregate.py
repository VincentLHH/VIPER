import sys
from tqdm import tqdm
from collections import defaultdict
import json
import joblib
import numpy as np
import math
def load_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            yield json.loads(line)

def save_features(feature_effect_set, feature_semantic_set, output_prefix="features"):
    effect_path = f"{output_prefix}_effect.pkl"
    semantic_path = f"{output_prefix}_semantic.pkl"

    joblib.dump(feature_effect_set, effect_path)
    joblib.dump(feature_semantic_set, semantic_path)

    print(f"Saved feature_effect_set to {effect_path}")
    print(f"Saved feature_semantic_set to {semantic_path}")

def extract_feature_effect(line):
    tag = line["model_output"]["logprobs"][0]["token"]
    emb = line["model_output"]["phrase_embeddings"]
    cpra = list(emb.keys())[-1]
    cpra_emb = np.array(emb[cpra]["mean_embedding"])
    return cpra, cpra_emb, tag


def extract_feature_semantic(line):
    tag = line["model_output"]["logprobs"][0]["token"]
    total_prob = sum(token_prob["logprob"] for token_prob in line["model_output"]["logprobs"])
    prob = math.exp(total_prob)
    # prob = math.exp(total_prob) if tag == "positive" else (1 - math.exp(total_prob))
    emb = line["model_output"]["phrase_embeddings"]
    cpra = list(emb.keys())[-1]
    feature_keys = list(emb.keys())[:-1]
    if not feature_keys:
        feature_emb = np.zeros(2048)
    else:
        embeddings = np.array([emb[key]["mean_embedding"] for key in feature_keys])  # shape: (N, 2048)
        importances = np.array([emb[key]["phrase_importance"] for key in feature_keys])  # shape: (N,)
        if importances.sum() > 0:
            weights = importances / importances.sum()
        else:
            weights = np.ones(len(importances)) / len(importances)

    feature_emb = weights @ embeddings
    return cpra, feature_emb, tag, prob

def main():
    input_ACMG = sys.argv[1]
    input_af = sys.argv[2]
    input_dbsig = sys.argv[3]
    input_dis = sys.argv[4]
    input_IUIS = sys.argv[5]
    input_pred = sys.argv[6]
    input_all = sys.argv[7]
    input_base = sys.argv[8]
    len = sys.argv[9]

    lines_ACMG = load_jsonl(input_ACMG)
    lines_af = load_jsonl(input_af)
    lines_dbsig = load_jsonl(input_dbsig)
    lines_dis = load_jsonl(input_dis)
    lines_IUIS = load_jsonl(input_IUIS)
    lines_pred = load_jsonl(input_pred)
    lines_all = load_jsonl(input_all)
    lines_base = load_jsonl(input_base)
    feature_effect_set = defaultdict(list)
    feature_semantic_set = defaultdict(list)

    print(f"Start extarcting features...")
    for line1, line2, line3, line4, line5, line6, line7, line8 in tqdm(zip(lines_ACMG, lines_af, lines_dbsig, lines_dis, lines_IUIS, lines_pred, lines_all, lines_base), total=len):
        cpra_ACMG, feature_emb_ACMG, tag_ACMG= extract_feature_effect(line1)
        cpra_ACMG_1, feature_semantic_ACMG, tag_semantic_ACMG, prob_ACMG = extract_feature_semantic(line1)
        cpra_af, feature_emb_af, tag_af = extract_feature_effect(line2)
        cpra_af_1, feature_semantic_af, tag_semantic_af, prob_af = extract_feature_semantic(line2)
        cpra_dbsig, feature_emb_dbsig, tag_dbsig = extract_feature_effect(line3)
        cpra_dbsig_1, feature_semantic_dbsig, tag_semantic_dbsig, prob_dbsig = extract_feature_semantic(line3)
        cpra_dis, feature_emb_dis, tag_dis = extract_feature_effect(line4)
        cpra_dis_1, feature_semantic_dis, tag_semantic_dis, prob_dis = extract_feature_semantic(line4)
        cpra_IUIS, feature_emb_IUIS, tag_IUIS = extract_feature_effect(line5)
        cpra_IUIS_1, feature_semantic_IUIS, tag_semantic_IUIS, prob_IUIS = extract_feature_semantic(line5)
        cpra_pred, feature_emb_pred, tag_pred = extract_feature_effect(line6)
        cpra_pred_1, feature_semantic_pred, tag_semantic_pred, prob_pred = extract_feature_semantic(line6)
        cpra_all, feature_emb_all, tag_all = extract_feature_effect(line7)
        cpra_all_1, feature_semantic_all, tag_semantic_all, prob_all = extract_feature_semantic(line7)
        cpra_base, feature_emb_base, tag_base = extract_feature_effect(line8)

        feature_effect = {
            'basic': feature_emb_base * 2,
            'ACMG': feature_emb_ACMG,
            'af': feature_emb_af,
            'dbsig': feature_emb_dbsig,
            'dis': feature_emb_dis,
            'IUIS': feature_emb_IUIS,
            'pred': feature_emb_pred,
            'all': feature_emb_all
        }
        for name, emb in feature_effect.items():
            effect = emb - feature_emb_base
            feature_effect_set[name].append(effect)

        feature_semantic = {
            'ACMG': feature_semantic_ACMG,
            'af': feature_semantic_af,
            'dbsig': feature_semantic_dbsig,
            'dis': feature_semantic_dis,
            'IUIS': feature_semantic_IUIS,
            'pred': feature_semantic_pred,
            'all': feature_semantic_all
        }
        for name, emb in feature_semantic.items():
            semantic = emb
            feature_semantic_set[name].append(semantic)

    save_features(feature_effect_set, feature_semantic_set, output_prefix="features")

if __name__ == "__main__":
    main()
