import json
import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

phrase_map = {
    "Clinvar database": "dbsig",
    "HGMD database": "dbsig",
    "ACMG evidence items": "ACMG",
    "Database assessment": "ACMG",
    "SIFT": "pred",
    "Polyphen2-HDIV": "pred",
    "Polyphen2-HVAR": "pred",
    "MutationTaster2": "pred",
    "Revel": "pred",
    "Condel": "pred",
    "SpliceAI": "pred",
    "dbscSnv_RF": "pred",
    "dbscSnv_ADA": "pred",
    "PhyloP Vertebrates": "pred",
    "PhyloP Placental Mammals": "pred",
    "GERP++": "pred",
    "Zygosity": "base",
    "Inheritance": "dis",
    "1000G": "af",
    "GnomAD": "af",
    "GnomAD-EAS": "af",
    "Disease JSON": "dis",
    "IEI/PID": "IUIS",
}
OUTPUT_FORMATS = ['png', 'svg', 'pdf']

def load_data(jsonl_path: str) -> List[Dict]:
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_category_attributions(data: List[Dict], use_abs_ig: bool = True) -> pd.DataFrame:
    records = []
    categories = set(phrase_map.values())

    for sample_id, item in enumerate(data):
        ig_scores_all = np.array(item["model_output"]["IG_scores"])

        if use_abs_ig:
            ig_scores_all = np.abs(ig_scores_all)

        response = item["model_output"]["response"]
        label = "positive" if "positive" in response else "negative"
        phrase_embs = item["model_output"]["phrase_embeddings"]

        cat_indices = {cat: [] for cat in categories}

        for phrase_key in phrase_embs.keys():
            matched = False
            for substr, cat in phrase_map.items():
                if substr in phrase_key:
                    indices = phrase_embs[phrase_key].get("token_indices", [])
                    valid_indices = [i for i in indices if i < len(ig_scores_all)]
                    cat_indices[cat].extend(valid_indices)
                    matched = True

        for cat in categories:
            if cat_indices[cat]:
                unique_indices = list(set(cat_indices[cat]))
                ig_vals = ig_scores_all[unique_indices].tolist()
                records.append({
                    "sample_id": sample_id,
                    "label": label,
                    "category": cat,
                    "token_indices": unique_indices,
                    "ig_scores": ig_vals
                })

    return pd.DataFrame(records)

def plot_distribution_all_and_by_category(df: pd.DataFrame, output_dir: str = "plots", save=False):
    os.makedirs(output_dir, exist_ok=True)
    categories = sorted(df['category'].unique())

    all_ig_scores = []
    for _, row in df.iterrows():
        all_ig_scores.extend(row['ig_scores'])
    all_ig_scores = np.array(all_ig_scores)
    low, high = np.percentile(all_ig_scores, 2.5), np.percentile(all_ig_scores, 97.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label in ['positive', 'negative']:
        subset = df[df['label'] == label]
        vals = []
        for _, row in subset.iterrows():
            vals.extend([x for x in row['ig_scores'] if low <= x <= high])
        sns.histplot(vals, kde=True, ax=ax, label=label, alpha=0.6, bins=50)
    ax.set_title("All Categories: IG Score Distribution by Label (2.5%-97.5% Clipped)")
    ax.set_xlabel("IG Score")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    if save:
        for fmt in OUTPUT_FORMATS:
            fig.savefig(os.path.join(output_dir, f"all_ig_distribution.{fmt}"), dpi=300, bbox_inches='tight')
    plt.show(fig)

    for cat in categories:
        subset = df[df['category'] == cat]
        if len(subset) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for label in ['positive', 'negative']:
            group = subset[subset['label'] == label]
            vals = []
            for _, row in group.iterrows():
                clipped = [x for x in row['ig_scores'] if low <= x <= high]
                vals.extend(clipped)
            if vals:
                sns.histplot(vals, kde=True, ax=ax, label=label, alpha=0.6, bins=50)
        ax.set_title(f"Category: {cat} - IG Score Distribution (Clipped)")
        ax.set_xlabel("IG Score")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()

        for fmt in OUTPUT_FORMATS:
            fig.savefig(os.path.join(output_dir, f"{cat}_ig_distribution.{fmt}"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved distribution plots for 'all' and {len(categories)} categories.")

def plot_category_comparison_with_test(df: pd.DataFrame, output_dir: str = "plots", save=True):
    os.makedirs(output_dir, exist_ok=True)
    categories = sorted(df['category'].unique())
    p_values = {}

    all_vals = np.array([x for _, row in df.iterrows() for x in row['ig_scores']])
    low, high = np.percentile(all_vals, 2.5), np.percentile(all_vals, 97.5)

    print("Processing 'all' category for comparison...")
    data_pos_all = []
    data_neg_all = []

    for _, row in df.groupby(['sample_id', 'label']):
        all_igs = []
        for _, r in row.iterrows():
            clipped = [x for x in r['ig_scores'] if low <= x <= high]
            all_igs.extend(clipped)
        mean_val = np.mean(all_igs) if all_igs else 0.0
        if row['label'].iloc[0] == 'positive':
            data_pos_all.append(mean_val)
        else:
            data_neg_all.append(mean_val)

    if len(data_pos_all) > 0 and len(data_neg_all) > 0:
        stat_all, p_val_all = mannwhitneyu(data_pos_all, data_neg_all, alternative='two-sided')
    else:
        p_val_all = 1.0
    p_values['all'] = p_val_all

    plot_data_all = pd.DataFrame({
        'Mean IG': data_pos_all + data_neg_all,
        'Label': ['positive'] * len(data_pos_all) + ['negative'] * len(data_neg_all)
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=plot_data_all, x='Label', y='Mean IG', ax=ax, palette="Set2")
    ax.set_title(f"All Categories - Mean IG by Label (p={p_val_all:.2e})")

    if len(plot_data_all) > 0:
        ymax = plot_data_all['Mean IG'].max()
        ymin = plot_data_all['Mean IG'].min()
        height = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
        ax.plot([0, 0, 1, 1], [ymax + height, ymax + 2*height, ymax + 2*height, ymax + height], lw=1.5, c="black")
        sig = "***" if p_val_all < 0.001 else "**" if p_val_all < 0.01 else "*" if p_val_all < 0.05 else "ns"
        ax.text(0.5, ymax + 2*height, sig, ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    for fmt in OUTPUT_FORMATS:
        fig.savefig(os.path.join(output_dir, f"all_violin_comparison.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    for cat in categories:
        subset = df[df['category'] == cat]
        if len(subset) == 0:
            continue

        data_pos = []
        data_neg = []

        for _, row in subset.iterrows():
            clipped = [x for x in row['ig_scores'] if low <= x <= high]
            mean_val = np.mean(clipped) if clipped else 0.0
            if row['label'] == 'positive':
                data_pos.append(mean_val)
            else:
                data_neg.append(mean_val)

        if len(data_pos) == 0 or len(data_neg) == 0:
            p_val = 1.0
        else:
            stat, p_val = mannwhitneyu(data_pos, data_neg, alternative='two-sided')
        p_values[cat] = p_val

        plot_data = pd.DataFrame({
            'Mean IG': data_pos + data_neg,
            'Label': ['positive'] * len(data_pos) + ['negative'] * len(data_neg)
        })
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=plot_data, x='Label', y='Mean IG', ax=ax, palette="Set2")
        ax.set_title(f"{cat} - Mean IG by Label (p={p_val:.2e})")

        if len(plot_data) > 0:
            ymax = plot_data['Mean IG'].max()
            ymin = plot_data['Mean IG'].min()
            height = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
            ax.plot([0, 0, 1, 1], [ymax + height, ymax + 2*height, ymax + 2*height, ymax + height], lw=1.5, c="black")
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax.text(0.5, ymax + 2*height, sig, ha='center', va='bottom', fontsize=12)

        plt.tight_layout()

        for fmt in OUTPUT_FORMATS:
            fig.savefig(os.path.join(output_dir, f"{cat}_violin_comparison.{fmt}"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("\nMann-Whitney U Test P-values by Category (including 'all'):")
    for cat in ['all'] + sorted(categories):
        if cat in p_values:
            print(f"{cat:15}: {p_values[cat]:.2e}")


def plot_pca_umap_by_attribution_vector(data: List[Dict], output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    categories = sorted(set(phrase_map.values()))
    feature_matrix = []
    labels = []

    for item in data:
        ig_scores_all = np.array(item["model_output"]["IG_scores"])
        label = "positive" if "positive" in item["model_output"]["response"] else "negative"
        labels.append(label)

        phrase_embs = item["model_output"]["phrase_embeddings"]
        cat_vals = {cat: [] for cat in categories}

        for phrase_key in phrase_embs:
            for substr, cat in phrase_map.items():
                if substr in phrase_key:
                    indices = phrase_embs[phrase_key].get("token_indices", [])
                    valid_indices = [i for i in indices if i < len(ig_scores_all)]
                    cat_vals[cat].extend(ig_scores_all[valid_indices])

        row = []
        for cat in categories:
            vals = cat_vals[cat]
            mean_val = np.mean(vals) if vals else 0.0
            row.append(mean_val)
        feature_matrix.append(row)

    X = np.array(feature_matrix)
    y = np.array(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Label': y})
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Label', ax=axes[0], alpha=0.8, s=60)
    axes[0].set_title(f'PCA - Explained Variance: {pca.explained_variance_ratio_.sum():.1%}')

    df_umap = pd.DataFrame({'UMAP1': X_umap[:, 0], 'UMAP2': X_umap[:, 1], 'Label': y})
    sns.scatterplot(data=df_umap, x='UMAP1', y='UMAP2', hue='Label', ax=axes[1], alpha=0.8, s=60)
    axes[1].set_title('UMAP')

    plt.tight_layout()
    for fmt in OUTPUT_FORMATS:
        fig.savefig(os.path.join(output_dir, f"ori_pca_umap_attribution_vector.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

def plot_pca_umap_from_category_df(df: pd.DataFrame, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    categories = sorted(df['category'].unique())
    sample_vectors = []
    sample_labels = []

    for (sample_id, label), group in df.groupby(['sample_id', 'label']):
        vector = []
        for cat in categories:
            subset = group[group['category'] == cat]
            if len(subset) == 0:
                mean_ig = 0.0
            else:
                all_ig_vals = []
                for ig_list in subset['ig_scores']:
                    all_ig_vals.extend(ig_list)
                mean_ig = np.mean(all_ig_vals) if all_ig_vals else 0.0
            vector.append(mean_ig)
        sample_vectors.append(vector)
        sample_labels.append(label)

    X = np.array(sample_vectors)
    y = np.array(sample_labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Label': y})
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Label', ax=axes[0], alpha=0.8, s=60)
    axes[0].set_title(f'PCA - Explained Variance: {pca.explained_variance_ratio_.sum():.1%}')

    df_umap = pd.DataFrame({'UMAP1': X_umap[:, 0], 'UMAP2': X_umap[:, 1], 'Label': y})
    sns.scatterplot(data=df_umap, x='UMAP1', y='UMAP2', hue='Label', ax=axes[1], alpha=0.8, s=60)
    axes[1].set_title('UMAP')

    plt.tight_layout()
    base_name = "pca_umap_from_category_vector"
    for fmt in OUTPUT_FORMATS:
        fig.savefig(os.path.join(output_dir, f"{base_name}.{fmt}"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

def run_full_analysis(jsonl_path: str, output_dir: str = "plots", use_abs_ig: bool = True, save = False):
    print(f"Loading data...")
    data = load_data(jsonl_path)

    mode_str = "abs_ig" if use_abs_ig else "raw_ig"
    output_dir = os.path.join(output_dir, mode_str)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting category-level attributions (use_abs_ig={use_abs_ig})...")
    df = extract_category_attributions(data, use_abs_ig=use_abs_ig)

    print("Plotting distribution (all + per category)...")
    plot_distribution_all_and_by_category(df, output_dir)

    print("Plotting category comparison with significance test...")
    plot_category_comparison_with_test(df, output_dir)

    print("Plotting PCA/UMAP of attribution vectors...")
    plot_pca_umap_by_attribution_vector(data, output_dir)
    plot_pca_umap_from_category_df(df, output_dir)

    print(f"All analysis completed. Plots saved to '{output_dir}'")


if __name__ == "__main__":
    JSONL_PATH = "IG_2342.jsonl"
    OUTPUT_DIR = "attribution_plots_v4"

    run_full_analysis(JSONL_PATH, OUTPUT_DIR, use_abs_ig=True)
