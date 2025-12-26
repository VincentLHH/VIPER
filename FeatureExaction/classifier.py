from sklearn.model_selection import train_test_split


def prepare_data(feature_collections, test_size=0.2, random_state=42):
    features = []
    labels = []

    for data_tag, embeddings in feature_collections.items():
        label = 1 if data_tag == 'S0' else 0
        features.extend(embeddings)
        labels.extend([label] * len(embeddings))

    X = np.array(features)
    y = np.array(labels)

    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    return X_train, X_test, y_train, y_test


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


def train_models(X_train, X_test, y_train, y_test, scale_features=False):

    models = {}
    y_proba_dict = {}

    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr
    y_proba_dict['Logistic Regression'] = lr.predict_proba(X_test_scaled)[:, 1]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    y_proba_dict['Random Forest'] = rf.predict_proba(X_test)[:, 1]

    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    models['Linear SVM'] = svm
    y_proba_dict['Linear SVM'] = svm.predict_proba(X_test_scaled)[:, 1]

    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    y_proba_dict['XGBoost'] = xgb.predict_proba(X_test)[:, 1]

    return models, y_proba_dict

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_prc_curves(fold_results, model_stats, filename, save=False):
    plt.figure(figsize=(8, 7))
    all_y_true = np.concatenate([y_true for folds in fold_results.values() for y_true, _ in folds])
    base_precision = np.mean(all_y_true)
    plt.axhline(y=base_precision, color='k', linestyle='--', alpha=0.7,
                label=f'Baseline (Precision={base_precision:.3f})')

    colors = cycle(['blue', 'red', 'green', 'orange'])
    mean_recall = np.linspace(0, 1.0, 100)

    for (model_name, folds), color in zip(fold_results.items(), colors):
        if not folds:
            continue

        precisions = []
        auprcs = []

        for y_true, y_proba in folds:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            auprc = average_precision_score(y_true, y_proba)
            auprcs.append(auprc)

            recall = recall[::-1]
            precision = precision[::-1]

            if precision[1] < 0.8:
                precision[0] = precision[1]

            interp_precision = np.interp(mean_recall, recall, precision)
            precisions.append(interp_precision)

        if not precisions:
            continue

        precisions = np.array(precisions)

        mean_precision = np.mean(precisions, axis=0)
        std_precision = np.std(precisions, axis=0)
        mean_recall_valid = mean_recall

        stats = model_stats[model_name]
        label = f'{model_name} (AUPRC = {stats["mean_auprc"]:.3f}±{stats["std_auprc"]:.3f})'
        plt.plot(mean_recall_valid, mean_precision, color=color, lw=2, label=label)

        precision_upper = np.minimum(mean_precision + std_precision, 1)
        precision_lower = np.maximum(mean_precision - std_precision, 0)
        plt.fill_between(mean_recall_valid, precision_lower, precision_upper, color=color, alpha=0.2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (Mean ± Std)', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        for ext in ['svg', 'png', 'pdf']:
            plt.savefig(f"{filename}_prc.{ext}", dpi=300, bbox_inches='tight')
    plt.show()


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

def lightweight_classifier(feature_collections, scale_features=False, random_state=42, filename="results", save=False):
    X_train_full, X_test_full, y_train_full, y_test_full = prepare_data(
        feature_collections, test_size=0.0, random_state=random_state
    )
    X, y = X_train_full, y_train_full
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    model_metrics = {
        'Logistic Regression': {'auc': [], 'auprc': [], 'brier': []},
        'Random Forest': {'auc': [], 'auprc': [], 'brier': []},
        'Linear SVM': {'auc': [], 'auprc': [], 'brier': []},
        'XGBoost': {'auc': [], 'auprc': [], 'brier': []}
    }

    fold_predictions = {model_name: [] for model_name in model_metrics.keys()}

    print(f"\nStart {n_splits}th training...\n")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold_idx + 1} ---")
        train_idx, test_idx = test_idx, train_idx
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        models, y_proba_dict = train_models(X_train, X_test, y_train, y_test, scale_features=scale_features)

        for model_name in model_metrics.keys():
            if model_name not in y_proba_dict:
                continue
            y_proba = y_proba_dict[model_name]

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = auc(fpr, tpr)
            auprc_val = average_precision_score(y_test, y_proba)
            brier_score = np.mean((y_test - y_proba) ** 2)

            model_metrics[model_name]['auc'].append(auc_val)
            model_metrics[model_name]['auprc'].append(auprc_val)
            model_metrics[model_name]['brier'].append(brier_score)

            fold_predictions[model_name].append((y_test, y_proba))

    def compute_mean_std(values):
        mean_val = np.mean(values)
        std_val = np.std(values)
        return mean_val, std_val

    model_stats = {}
    for model_name in model_metrics:
        mean_auc, std_auc = compute_mean_std(model_metrics[model_name]['auc'])
        mean_auprc, std_auprc = compute_mean_std(model_metrics[model_name]['auprc'])
        mean_brier, std_brier = compute_mean_std(model_metrics[model_name]['brier'])
        model_stats[model_name] = {
            'mean_auc': mean_auc, 'std_auc': std_auc,
            'mean_auprc': mean_auprc, 'std_auprc': std_auprc,
            'mean_brier': mean_brier, 'std_brier': std_brier
        }

    plot_prc_curves(fold_predictions, model_stats, f"{filename}", save=save)

    return model_stats
