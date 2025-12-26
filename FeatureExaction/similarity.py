import numpy as np
from collections import defaultdict
import pickle
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score
import joblib
import numpy as np
import pandas as pd
import joblib
from scipy import stats

def train_mapping_model(X, y, model_type='linear', cv=5):
    """
    Train a model to map from X (semantic) to y (influence) with cross-validation.
    X, y: numpy arrays of shape (N, dim)
    model_type: 'linear' for LinearRegression, 'mlp' for MLPRegressor
    cv: number of folds for cross-validation
    Returns: average R2 score across folds
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(512,), max_iter=100, random_state=42)
    else:
        raise ValueError("Unsupported model_type. Use 'linear' or 'mlp'.")

    # Use cross_val_score for CV R2
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    avg_r2 = np.mean(scores)
    std_r2 = np.std(scores, ddof=1)
    return avg_r2, std_r2

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        n = gram.shape[0]
        if n <= 3:
            return np.zeros_like(gram)
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, axis=0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None] + means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, axis=0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None] + means[None, :]
    return gram

def cka(X, Y, kernel='linear', unbiased=False, **kernel_kwargs):
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"

    if kernel == 'linear':
        K = gram_linear(X)
        L = gram_linear(Y)
    elif kernel == 'rbf':
        K = gram_rbf(X, **kernel_kwargs)
        L = gram_rbf(Y, **kernel_kwargs)
        # from sklearn.metrics.pairwise import pairwise_kernels
        # K = pairwise_kernels(X, metric=kernel, **kernel_kwargs)
        # L = pairwise_kernels(Y, metric=kernel, **kernel_kwargs)
    else:
        # fallback to sklearn for other kernels
        from sklearn.metrics.pairwise import pairwise_kernels
        K = pairwise_kernels(X, metric=kernel, **kernel_kwargs)
        L = pairwise_kernels(Y, metric=kernel, **kernel_kwargs)

    Kc = center_gram(K, unbiased=unbiased)
    Lc = center_gram(L, unbiased=unbiased)

    hsic = np.sum(Kc * Lc)
    norm_K = np.linalg.norm(Kc)
    norm_L = np.linalg.norm(Lc)

    return hsic / (norm_K * norm_L)


def average_cka_over_subsets(X, Y, subset_size=500, num_subsets=100,
                             confidence=0.95,
                             kernel='linear',
                             seed=42,
                             unbiased=False,
                             **kernel_kwargs):
    """
    Compute average CKA over bootstrapped subsets with optional kernel.
    """
    N = X.shape[0]
    cka_scores = []
    rng = np.random.default_rng(seed=seed)
    for _ in range(num_subsets):
        idx = rng.choice(N, subset_size, replace=False)
        X_sub = X[idx]
        Y_sub = Y[idx]
        cka_val = cka(X_sub, Y_sub, kernel=kernel, unbiased= unbiased, **kernel_kwargs)
        cka_scores.append(cka_val)

    cka_scores = np.array(cka_scores)
    mean_cka = np.mean(cka_scores)
    std = np.std(cka_scores, ddof=1)

    # if num_subsets > 1:
    #     alpha = 1 - confidence
    #     lower_percentile = 100 * (alpha / 2)
    #     upper_percentile = 100 * (1 - alpha / 2)
    #     ci_lower = np.percentile(cka_scores, lower_percentile)
    #     ci_upper = np.percentile(cka_scores, upper_percentile)
    # else:
    #     ci_lower = ci_upper = mean_cka

    return mean_cka, std


def compute_cosine_similarity_with_stats(semantic_vecs, effect_vecs):
    """
    Compute cosine similarity between two sets of vectors (aligned by sample index),
    and test whether the mean cosine similarity is significantly different from 0.

    Parameters:
        semantic_vecs : np.ndarray of shape (N, D)
        effect_vecs   : np.ndarray of shape (M, D)

    Returns:
        dict with keys:
            'cos_mean', 'cos_p_value', 'cos_cohens_d', 'n_samples'
    """
    n_min = min(len(semantic_vecs), len(effect_vecs))
    if n_min == 0:
        return {
            'cos_mean': np.nan,
            'cos_p_value': np.nan,
            'cos_cohens_d': np.nan,
            'n_samples': 0
        }

    sem_sub = semantic_vecs[:n_min]
    eff_sub = effect_vecs[:n_min]

    # Normalize to avoid recomputing norms repeatedly
    norms_sem = np.linalg.norm(sem_sub, axis=1, keepdims=True)
    norms_eff = np.linalg.norm(eff_sub, axis=1, keepdims=True)

    # Avoid division by zero
    norms_sem = np.where(norms_sem == 0, 1.0, norms_sem)
    norms_eff = np.where(norms_eff == 0, 1.0, norms_eff)

    cos_sim = np.sum(sem_sub * eff_sub / (norms_sem * norms_eff), axis=1)
    n = len(cos_sim)
    cos_mean = np.mean(cos_sim)
    cos_std = np.std(cos_sim, ddof=1)

    if n < 2 or cos_std == 0:
        p_val = np.nan
        cohens_d = np.nan
    else:
        # Two-tailed one-sample t-test against 0
        _, p_val = stats.ttest_1samp(cos_sim, popmean=0.0)
        cohens_d = cos_mean / cos_std  # One-sample Cohen's d

    return {
        'cos_mean': cos_mean,
        'cos_p_value': p_val,
        'cos_cohens_d': cohens_d,
        'cos_std': cos_std,
        'n_samples': n
    }


# - average_cka_over_subsets(...)
# - train_mapping_model(X, y, model_type, cv) -> (mean_r2, std_r2)

def run_full_feature_analysis_v2(
    file_path_effect='features/features_effect.pkl',
    file_path_semantic='features/features_semantic.pkl',
    subset_size=10000,
    num_subsets=100,
    kernel='linear',
    unbiased=False,
    mapping_model_type='linear',
    mapping_cv=5,
    **kernel_kwargs
):

    print("Loading feature sets...")
    feature_effect_set = joblib.load(file_path_effect)
    feature_semantic_set = joblib.load(file_path_semantic)

    effect_labels = sorted(feature_effect_set.keys())
    semantic_labels = sorted(feature_semantic_set.keys())

    assert set(effect_labels) == set(semantic_labels), "Labels in effect and semantic sets must be identical!"
    print(f"Total labels: {len(effect_labels)} | Performing full cross analysis ({len(effect_labels)} x {len(semantic_labels)})")

    results = []

    for label_i in effect_labels:
        inf_vecs = np.array(feature_effect_set[label_i])
        if len(inf_vecs) == 0 or np.isnan(inf_vecs).any() or np.isinf(inf_vecs).any():
            print(f"Skipping effect label '{label_i}': invalid data.")
            continue

        for label_j in semantic_labels:
            sem_vecs = np.array(feature_semantic_set[label_j])
            if len(sem_vecs) == 0 or np.isnan(sem_vecs).any() or np.isinf(sem_vecs).any():
                print(f"Skipping pair ({label_i}, {label_j}): semantic data invalid.")
                # Append NaN row
                cos_mean = cos_p = cos_cohens_d = cos_std = np.nan
                cka_score = cka_std = np.nan
                r2_inf2sem = r2_sem2inf = std_inf2sem = std_sem2inf = np.nan
            else:
                n_min = min(len(inf_vecs), len(sem_vecs))
                inf_sub = inf_vecs[:n_min]
                sem_sub = sem_vecs[:n_min]

                # --- 1. Cosine Similarity (vs 0 baseline) ---
                cos_res = compute_cosine_similarity_with_stats(sem_sub, inf_sub)
                cos_mean = cos_res['cos_mean']
                cos_p = cos_res['cos_p_value']
                cos_cohens_d = cos_res['cos_cohens_d']
                cos_std = cos_res['cos_std']

                # --- 2. CKA ---
                cka_score, cka_std = average_cka_over_subsets(
                    sem_sub, inf_sub,
                    subset_size=subset_size,
                    num_subsets=num_subsets,
                    kernel=kernel,
                    unbiased=unbiased,
                    **kernel_kwargs
                )

                # --- 3. Bidirectional Linear Mapping (full cross) ---
                # inf[i] -> sem[j]
                try:
                    r2_inf2sem, std_inf2sem = train_mapping_model(inf_sub, sem_sub, mapping_model_type, mapping_cv)
                except Exception as e:
                    print(f"Mapping failed for inf[{label_i}] → sem[{label_j}]: {e}")
                    r2_inf2sem = std_inf2sem = np.nan

                # sem[i] -> inf[j]
                inf_j = np.array(feature_effect_set[label_j])
                sem_i = np.array(feature_semantic_set[label_i])
                if len(inf_j) == 0 or len(sem_i) == 0:
                    r2_sem2inf = std_sem2inf = np.nan
                else:
                    n_min2 = min(len(sem_i), len(inf_j))
                    sem_i_sub = sem_i[:n_min2]
                    inf_j_sub = inf_j[:n_min2]
                    try:
                        r2_sem2inf, std_sem2inf = train_mapping_model(sem_i_sub, inf_j_sub, mapping_model_type, mapping_cv)
                    except Exception as e:
                        print(f"Mapping failed for sem[{label_i}] → inf[{label_j}]: {e}")
                        r2_sem2inf = std_sem2inf = np.nan

            results.append({
                'Effect_Label': label_i,
                'Semantic_Label': label_j,
                'Cosine_Mean': cos_mean,
                'Cosine_p_value': cos_p,
                'Cosine_Cohens_d': cos_cohens_d,
                'Cosine_std': cos_std,
                'CKA_Score': cka_score,
                'STDE_CKA': cka_std,
                'R2_Inf_to_Sem': r2_inf2sem,          # effect[i] → semantic[j]
                'R2_Sem_to_Inf': r2_sem2inf,          # semantic[i] → effect[j]
                'Std_R2_Inf_to_Sem': std_inf2sem,
                'Std_R2_Sem_to_Inf': std_sem2inf,
            })

            print(f" {label_i} vs {label_j} | "
                  f"Cos: {cos_mean:.4f} (p={cos_p:.3f}, d={cos_cohens_d:.3f}) | "
                  f"CKA: {cka_score:.4f} | "
                  f"R2: {r2_inf2sem:.4f} / {r2_sem2inf:.4f}")

    df = pd.DataFrame(results)
    return df, feature_effect_set, feature_semantic_set

df,_,_= run_full_feature_analysis_v2()
df.to_csv('example_full_sim.csv', index=False)