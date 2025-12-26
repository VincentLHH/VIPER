import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import TSNE


def load_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            yield json.loads(line)


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
        # 归一化重要性权重
        if importances.sum() > 0:
            weights = importances / importances.sum()
        else:
            weights = np.ones(len(importances)) / len(importances)

    # 向量化加权求和: (N,) @ (N, 2048) -> (2048,)
    feature_emb = weights @ embeddings
    return cpra, feature_emb, tag, prob


def prepare_clustering_data(tagged_features, target_tags=None):
    if target_tags is None:
        target_tags = list(tagged_features.keys())
    all_vectors = []
    all_labels = []
    for tag in target_tags:
        if tag in tagged_features:
            vectors = tagged_features[tag]
            n_samples = vectors.shape[0]

            all_vectors.append(vectors)
            all_labels.extend([tag] * n_samples)
    X = np.vstack(all_vectors)  # 合并所有样本
    y = np.array(all_labels)  # 对应的标签
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def analyze_intra_class_similarity(feature_collections,
                                   normalize='l2',
                                   plot='hist',
                                   verbose=True):
    """
    对 feature_collections 中每个标签计算类内特征相似度（余弦相似度）

    Parameters:
    -----------
    feature_collections : defaultdict(list) 或 dict
        键为标签名，值为特征向量列表，每个向量 shape 应为 (feature_dim,)
    normalize : str, default='standardize'
        标准化方式:
        - 'standardize': Z-score (均值0，方差1)
        - 'l2': L2 归一化（适合余弦相似度）
        - None: 不标准化
    plot : bool, default=False
        是否绘制相似度分布图或热图
    verbose : bool, default=True
        是否打印结果

    Returns:
    --------
    results : dict
        每个标签的相似度统计信息，如：
        {
            'cat': {'mean': 0.85, 'std': 0.05, 'size': 10},
            'dog': {'mean': 0.78, 'std': 0.08, 'size': 12},
            ...
        }
    """
    results = {}

    # 支持的归一化方式
    if normalize not in ['standardize', 'l2', None]:
        raise ValueError("normalize must be 'standardize', 'l2', or None")

    # 收集所有标签的平均相似度，用于最后绘图
    labels = []
    means = []

    for label, feats in feature_collections.items():
        if len(feats) < 2:
            print(f"跳过标签 '{label}': 样本数 < 2，无法计算相似度")
            continue

        # 转为 numpy 数组 (N, D)
        X = np.array(feats)
        N, D = X.shape

        # 标准化
        if normalize == 'standardize':
            X = StandardScaler().fit_transform(X)
        elif normalize == 'l2':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-8)  # 防除零

        # 计算余弦相似度矩阵
        sim_matrix = cosine_similarity(X)  # (N, N)

        # 提取上三角（避免重复和对角线）
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        intra_sim = sim_matrix[triu_indices]

        mean_sim = np.mean(intra_sim)
        std_sim = np.std(intra_sim)

        results[label] = {
            'mean': mean_sim,
            'std': std_sim,
            'size': N,
            'min': np.min(intra_sim),
            'max': np.max(intra_sim),
            'values': intra_sim
        }

        labels.append(label)
        means.append(mean_sim)

        if verbose:
            print(f"标签 '{label}':")
            print(f"样本数: {N}, 特征维数: {D}")
            print(f"平均类内相似度: {mean_sim:.4f} ± {std_sim:.4f}")
            print(f"范围: [{np.min(intra_sim):.4f}, {np.max(intra_sim):.4f}]")
            print("-" * 50)

    # 绘图
    if plot and labels:
        plt.figure(figsize=(10, 6))

        if plot == 'hist':
            # 相似度分布直方图（所有类合并）
            all_sims = np.concatenate([results[label]['values'] for label in results])
            sns.histplot(all_sims, kde=True, bins=30)
            plt.title("类内相似度分布（所有类别）")
            plt.xlabel("余弦相似度")
            plt.ylabel("频次")

        elif plot == 'heatmap':
            # 相似度矩阵热图（只显示第一个类）
            first_label = list(results.keys())[0]
            X = np.array(feature_collections[first_label])
            if normalize == 'standardize':
                X = StandardScaler().fit_transform(X)
            elif normalize == 'l2':
                X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            sim_matrix = cosine_similarity(X)
            sns.heatmap(sim_matrix, annot=False, cmap='Blues', cbar=True)
            plt.title(f"标签 '{first_label}' 的类内相似度热图")
            plt.xlabel("样本索引")
            plt.ylabel("样本索引")

        else:
            # 默认：条形图显示每个类的平均相似度
            df = pd.DataFrame({'Label': labels, 'Mean Similarity': means})
            sns.barplot(data=df, x='Label', y='Mean Similarity')
            plt.title("各类的平均类内相似度")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)

        plt.tight_layout()
        plt.show()
        plt.savefig('intra_class_similarity.pdf')
        plt.savefig('intra_class_similarity.svg')
        plt.savefig('intra_class_similarity.png')

    return results





def analyze_cross_class_similarity_per_sample(feature_collections,
                                              label1, label2,
                                              normalize="l2",
                                              plot=True,
                                              verbose=True,
                                              mean=0.5):
    """
    计算同一样本在两个类别（label1 vs label2）下的特征之间的余弦相似度

    Parameters:
    -----------
    feature_collections : dict or defaultdict
        feature_collections[label1] 和 [label2] 应包含按顺序对应的样本特征列表
    label1, label2 : str
        要比较的两个类别标签
    normalize : str, default=None
        标准化方式: 'standardize', 'l2', 或 None
    plot : bool or str, default=False
        是否绘图: True/'dist' 显示分布, 'qq' 显示Q-Q图
    verbose : bool, default=True
        是否打印结果

    Returns:
    --------
    results : dict
        包含相似度数组、均值、统计检验结果等
    """
    if label1 not in feature_collections or label2 not in feature_collections:
        raise ValueError(f"Labels {label1} or {label2} not in feature_collections")

    feats1 = feature_collections[label1]  # list of arrays
    feats2 = feature_collections[label2]

    if len(feats1) != len(feats2):
        raise ValueError(f"样本数不匹配: {label1}({len(feats1)}) vs {label2}({len(feats2)})")

    n_samples = len(feats1)
    if n_samples == 0:
        raise ValueError("没有样本")

    # 转为 numpy 数组 (N, D)
    X1 = np.array([f if isinstance(f, np.ndarray) else np.array(f) for f in feats1])
    X2 = np.array([f if isinstance(f, np.ndarray) else np.array(f) for f in feats2])

    if X1.shape[1] != X2.shape[1]:
        raise ValueError(f"特征维度不匹配: {X1.shape[1]} vs {X2.shape[1]}")

    D = X1.shape[1]

    # 标准化（可选）
    if normalize == 'standardize':
        scaler = StandardScaler()
        X1 = scaler.fit_transform(X1)
        X2 = scaler.transform(X2)  # 使用 label1 的 scaler
    elif normalize == 'l2':
        X1 = X1 / (np.linalg.norm(X1, axis=1, keepdims=True) + 1e-8)
        X2 = X2 / (np.linalg.norm(X2, axis=1, keepdims=True) + 1e-8)

    # 计算每个样本的余弦相似度（逐样本）
    similarities = []
    for i in range(n_samples):
        # reshape 为 (1, D) 以兼容 cosine_similarity
        sim = cosine_similarity(X1[i].reshape(1, -1), X2[i].reshape(1, -1))[0][0]
        similarities.append(sim)

    similarities = np.array(similarities)

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    sem_sim = std_sim / np.sqrt(n_samples)  # 标准误

    # ✅ 统计学分析：假设检验
    # H0: 平均相似度 = 0.5（举例），H1: > 0.5
    # 使用单样本 t-test（假设相似度近似正态）
    t_stat, p_value = stats.ttest_1samp(similarities, popmean=mean)

    # 95% 置信区间（t 分布）
    if n_samples > 1:
        ci = stats.t.interval(0.95, df=n_samples - 1, loc=mean_sim, scale=sem_sim)
    else:
        ci = (mean_sim, mean_sim)

    results = {
        'label1': label1,
        'label2': label2,
        'n_samples': n_samples,
        'feature_dim': D,
        'similarities': similarities,
        'mean': mean_sim,
        'std': std_sim,
        'sem': sem_sim,
        'ci_95': ci,
        't_stat': t_stat,
        'p_value': p_value,
        'effect_size_cohen_d': (mean_sim - mean) / std_sim if std_sim > 0 else 0
    }

    if verbose:
        print(f"同一样本在 '{label1}' vs '{label2}' 下的特征相似度分析")
        print(f"样本数: {n_samples}, 特征维数: {D}")
        print(f"平均余弦相似度: {mean_sim:.4f} ± {std_sim:.4f} (mean ± std)")
        print(f"95% 置信区间: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"单样本 t 检验 (H0: μ={mean}): t={t_stat:.3f}, p={p_value:.3e}")
        if p_value < 0.05:
            print(f"结果显著：拒绝 H0，平均相似度显著 ≠ {mean}")
        else:
            print(f"不显著：无法拒绝 H0")
        print(f" Cohen's d 效应量: {results['effect_size_cohen_d']:.3f}")

    # 绘图
    if plot and n_samples > 0:
        plt.figure(figsize=(10, 6))

        if plot == 'qq':
            # Q-Q 图检验正态性
            stats.probplot(similarities, dist="norm", plot=plt)
            plt.title(f"Q-Q plot: {label1} vs {label2} Cosine Similarity")
        else:
            # 默认：分布直方图 + KDE
            sns.histplot(similarities, kde=True, bins=min(20, n_samples // 2), alpha=0.6)
            plt.axvline(mean_sim, color='red', linestyle='--', label=f'Mean = {mean_sim:.3f}')
            plt.axvline(mean, color='gray', linestyle=':', label=f'Null μ={mean}')
            plt.title(f"Same Sample '{label1}' vs '{label2}' Cosine Sim Distribution")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Frequency")
            plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig(f"{label1}_vs_{label2}_similarity_dist.pdf")
        plt.savefig(f"{label1}_vs_{label2}_similarity_dist.svg")
        plt.savefig(f"{label1}_vs_{label2}_similarity_dist.png")

    return results





def analyze_feature_norm_vs_prob(feature_dict,
                                 prob_key='prob',
                                 exclude_keys=None,
                                 plot=True,
                                 method='pearson',
                                 verbose=True,
                                 confidence=False,
                                 log_transform=False):
    """
    分析每个类别特征的模长（L2 norm）与预测概率 'prob' 的相关性

    Parameters:
    -----------
    feature_dict : dict
        key: 标签名（如 'effect', 'semantic'），value: 特征列表（每个元素是 np.array 或 list）
        特别地，feature_dict[prob_key] 是预测概率列表
    prob_key : str, default='prob'
        存储预测概率的键名
    exclude_keys : list, optional
        要排除的键（如不参与分析的标签）
    plot : bool, default=True
        是否绘制散点图
    method : str, default='pearson'
        相关性方法: 'pearson' 或 'spearman'
    verbose : bool, default=True
        是否打印结果

    Returns:
    --------
    results : dict
        每个标签的模长、相关系数、p 值等
    """
    if prob_key not in feature_dict:
        raise ValueError(f"feature_dict 中必须包含 '{prob_key}' 键")

    # 获取 prob 并转为 numpy
    probs = np.array([p if np.isscalar(p) else p[0] if isinstance(p, (list, np.ndarray)) else p
                      for p in feature_dict[prob_key]])
    n_samples = len(probs)

    if verbose:
        print(f"分析特征模长与预测概率 '{prob_key}' 的相关性")
        print(f"样本数: {n_samples}")

    # 排除不需要的键
    keys_to_analyze = [k for k in feature_dict.keys()
                       if k != prob_key and (exclude_keys is None or k not in exclude_keys)]

    results = {}
    valid_keys = []

    for label in keys_to_analyze:
        if label not in feature_dict:
            continue

        feats = feature_dict[label]
        if len(feats) != n_samples:
            print(f"跳过 '{label}': 样本数不匹配 ({len(feats)} vs {n_samples})")
            continue

        # 提取每个特征的 L2 模长
        norms = []
        for feat in feats:
            feat = np.array(feat)
            if feat.ndim == 0:
                norm = float(feat)
            else:
                norm = np.linalg.norm(feat.flatten())
            norms.append(norm)
        norms = np.array(norms)
        if log_transform:
            norms = np.log(norms + 1e-8)

        if confidence:
            norms = 1 / (1 + norms)

        # 计算相关性
        if method == 'pearson':
            corr, p_val = pearsonr(norms, probs)
            corr_type = "Pearson"
        else:
            corr, p_val = spearmanr(norms, probs)
            corr_type = "Spearman (rank)"

        results[label] = {
            'norms': norms,
            'probs': probs,
            'correlation': corr,
            'p_value': p_val,
            'n_samples': n_samples,
            'mean_norm': np.mean(norms),
            'mean_prob': np.mean(probs)
        }

        valid_keys.append(label)

        if verbose:
            print(f"\n标签 '{label}':")
            print(f"平均模长: {np.mean(norms):.4f}, 平均概率: {np.mean(probs):.4f}")
            print(f"{corr_type} 相关性: {corr:.4f} (p = {p_val:.3e})")
            if p_val < 0.05:
                print(f"显著相关" + ("（正）" if corr > 0 else "（负）"))
            else:
                print(f"不显著相关")

    # 绘图
    if plot and valid_keys:
        n_cols = 2
        n_rows = (len(valid_keys) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]  # 单子图情况
        else:
            axes = axes.flatten()  # 多子图，直接展平

        for idx, label in enumerate(valid_keys):
            ax = axes[idx]
            norms = results[label]['norms']
            probs = results[label]['probs']
            corr = results[label]['correlation']
            p_val = results[label]['p_value']

            # sns.scatterplot(x=norms, y=probs, ax=ax, alpha=0.6)
            sns.regplot(x=norms, y=probs, ax=ax, scatter_kws={'color': 'blue', 'alpha': 0.6},
                        line_kws={'color': 'red', 'linewidth': 2})
            if log_transform:
                ax.set_title(f"{label} | Log Norm vs '{prob_key}'\n{method.title()} r = {corr:.3f}, p = {p_val:.1e}")
                ax.set_xlabel("Feature L2 Norm (Log scale)")
            else:
                ax.set_title(f"{label} | Norm vs '{prob_key}'\n{method.title()} r = {corr:.3f}, p = {p_val:.1e}")
                ax.set_xlabel("Feature L2 Norm")
            ax.set_ylabel(f"'{prob_key}' Value")

        # 隐藏多余的子图
        for idx in range(len(valid_keys), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()
        plt.savefig(f"{label}_norm_vs_{prob_key}_similarity_dist_{method}.pdf")
        plt.savefig(f"{label}_norm_vs_{prob_key}_similarity_dist_{method}.svg")
        plt.savefig(f"{label}_norm_vs_{prob_key}_similarity_dist_{method}.png")

    return results


def filter_visible_points(x, y, resolution=3000):
    """
    基于像素网格过滤：每个像素只保留最后一个点
    x, y 为 PCA 坐标
    resolution 控制网格大小（越大越保留更多点）
    """
    gx = np.floor((x - x.min()) / (x.max() - x.min()) * (resolution - 1)).astype(int)
    gy = np.floor((y - y.min()) / (y.max() - y.min()) * (resolution - 1)).astype(int)

    visible = {}
    for i in range(len(x)):
        visible[(gx[i], gy[i])] = i

    idx = list(visible.values())
    return idx


def plot_pca(X, y, title="PCA", continuous=False):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    y = np.array(y)
    if not continuous:
        label_order = list(dict.fromkeys(y))  # 按出现顺序记录
        y = pd.Categorical(y, categories=label_order, ordered=True)

    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    plt.figure(figsize=(8, 6))
        
    CB = [
    "#0072B2", "#D55E00", "#009E73",
    "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"
    ]   

    sns.set_palette(CB)

    if continuous:
        plt.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            c=y,
            cmap='viridis',
            s=20,  # 点更小更适合密集图
            rasterized=True
        )
    else:
        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=y,
            palette=CB,
            hue_order=label_order,
            s=20,
            rasterized=True
        )

    plt.title(f"{title}\n"
              f"PC1: {pca.explained_variance_ratio_[0]:.1%}, "
              f"PC2: {pca.explained_variance_ratio_[1]:.1%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    if not continuous:
        plt.legend(title="Label")
    plt.grid(False)

    plt.savefig(f"{title}.pdf")
    plt.savefig(f"{title}.svg")
    plt.savefig(f"{title}.png")
    plt.show()



def plot_pca_back(X, y, title="PCA", continuous=False):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.rcParams["font.family"]="DejaVu Serif"
    plt.rcParams["pdf.fonttype"]= 42
    if continuous:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            c=y,  # 使用连续变量作为颜色
            cmap='viridis',  # 使用适合连续数据的颜色映射
            s=50
        )
    else:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=y,
            palette='Set1',
            s=50
        )

    plt.title(f"{title}\n"
              f"PC1: {pca.explained_variance_ratio_[0]:.1%}, "
              f"PC2: {pca.explained_variance_ratio_[1]:.1%}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Label")
    plt.grid(False)
    plt.show()
    plt.savefig(f"{title}.pdf")
    plt.savefig(f"{title}.svg")
    plt.savefig(f"{title}.png")


def plot_umap(X, y, title="UMAP", n_components=2, random_state=42):
    """
    使用 UMAP 对高维数据进行降维并可视化。

    Parameters:
    - X: 特征矩阵 (n_samples, n_features)
    - y: 标签向量 (n_samples,)
    - title: 图表标题
    - n_components: 降维后的维度（默认为2）
    - random_state: 随机种子，保证结果可复现
    """
    # 执行 UMAP 降维
    umap = UMAP(
        n_components=n_components,
        n_neighbors=60,  # 控制局部 vs 全局结构。小 → 关注局部，大 → 关注全局
        min_dist=0.3,  # 点之间最小距离。小 → 聚类紧凑，大 → 聚类分散
        metric='cosine',  # 可尝试'euclidean', 'cosine', 'manhattan' 等
        random_state=random_state,  # 可复现
        spread=1.0,  # 与 min_dist 配合控制分布松散程度
        local_connectivity=1.0,
        repulsion_strength=1.0,
    )
    X_umap = umap.fit_transform(X)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=y,
        palette='Set1',
        s=60,
        alpha=0.8,
        legend='full'
    )

    plt.title(f"{title}")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Label")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title}.pdf")
    plt.savefig(f"{title}.svg")
    plt.savefig(f"{title}.png")


def plot_tsne(X, y, title="t-SNE",
              perplexity=30,
              learning_rate=200,
              n_iter=1000,
              random_state=42):
    """
    使用 t-SNE 对高维数据降维并可视化。

    Parameters:
    - X: 已标准化的特征矩阵 (n_samples, n_features)
    - y: 标签向量或列表
    - title: 图表标题
    - perplexity: t-SNE 的困惑度，控制局部/全局平衡，建议 5~50
    - learning_rate: 学习率，too low: 收敛慢，too high: 跳过结构
    - n_iter: 最大迭代次数
    - random_state: 随机种子，保证可复现
    """
    # 执行 t-SNE 降维
    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=random_state,
                init='pca',  # 收敛更稳定
                verbose=0)  # 不打印过程

    X_tsne = tsne.fit_transform(X)

    # 绘图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=y,
        palette='Set1',
        s=60,
        alpha=0.8
    )

    plt.title(f"{title}\n"
              f"Perplexity={perplexity}, LR={learning_rate}, Iter={n_iter}")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.legend(title="Label")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title}.pdf")
    plt.savefig(f"{title}.svg")
    plt.savefig(f"{title}.png")


def plot_pca_then_umap(X, y, n_pca_components=300, title="PCA + UMAP", random_state=42):
    """
    先对数据进行PCA降维，再在PCA结果上应用UMAP降维至2D并可视化。

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        输入的数据矩阵
    y : array-like, shape (n_samples,)
        标签数组，用于着色
    n_pca_components : int, default=50
        PCA保留的主成分数量（建议50或更高，取决于数据）
    title : str
        图表标题
    random_state : int
        UMAP的随机种子，保证可重现性
    """
    # Step 1: PCA降维
    print(f"Applying PCA to reduce to {n_pca_components} components...")
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    # 计算PCA累计解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    total_var = explained_variance_ratio.sum()

    # Step 2: 在PCA结果上应用UMAP
    print("Applying UMAP on PCA output...")
    umap_ = UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
    X_umap = umap_.fit_transform(X_pca)

    # Step 3: 可视化
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=y,
        palette='Set1',
        s=60,
        alpha=0.8,
        edgecolor=None
    )

    plt.title(f"{title}\n"
              f"PCA ({n_pca_components} comps, {total_var:.1%} variance explained) → "
              f"UMAP")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Label")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title}.pdf")
    plt.savefig(f"{title}.svg")
    plt.savefig(f"{title}.png")

    return X_pca, X_umap, pca, umap_
