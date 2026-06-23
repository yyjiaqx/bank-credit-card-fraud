"""
Credit Card Fraud Detection - Evaluation
Model evaluation, metrics, visualization, feature importance, and SHAP.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, recall_score, precision_score
)
import os

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 120
plt.rcParams['font.size'] = 11


def evaluate_model(model, model_name, X_test, y_test, output_dir="outputs"):
    """Evaluate a trained model and generate plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\n{'=' * 50}")
    print(f"Evaluation: {model_name}")
    print(f"{'=' * 50}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  PR-AUC:    {auprc:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

    return {
        'Model': model_name,
        'ROC-AUC': auc,
        'PR-AUC': auprc,
        'Recall': recall,
        'Precision': precision,
        'F1-Score': f1,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


def plot_confusion_matrix(cm, model_name, output_dir):
    """Plot and save a single confusion matrix."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    path = os.path.join(output_dir, f'cm_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(path)
    plt.close()


def plot_roc_curves(models_dict, X_test, y_test, output_dir="outputs"):
    """Plot all ROC curves on one figure."""
    plt.figure(figsize=(8, 6))
    colors = {'Logistic Regression': '#e74c3c',
              'Random Forest': '#2ecc71',
              'XGBoost': '#3498db'}
    for name, (model, _) in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})',
                 color=colors.get(name, '#333'), linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    print(f"[INFO] ROC curves saved")


def plot_pr_curves(models_dict, X_test, y_test, output_dir="outputs"):
    """Plot all PR curves on one figure."""
    plt.figure(figsize=(8, 6))
    colors = {'Logistic Regression': '#e74c3c',
              'Random Forest': '#2ecc71',
              'XGBoost': '#3498db'}
    baseline = np.sum(y_test) / len(y_test)
    for name, (model, _) in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        pr, rec, _ = precision_recall_curve(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        plt.plot(rec, pr, label=f'{name} (AP={auprc:.4f})',
                 color=colors.get(name, '#333'), linewidth=2)
    plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7,
                label=f'Baseline ({baseline:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'))
    plt.close()
    print("[INFO] PR curves saved")


def plot_model_comparison(metrics_list, output_dir="outputs"):
    """Bar chart comparing all models across metrics."""
    df = pd.DataFrame(metrics_list).set_index('Model')
    metrics_cols = ['ROC-AUC', 'PR-AUC', 'Recall', 'Precision', 'F1-Score']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.15
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

    for i, (col, color) in enumerate(zip(metrics_cols, colors)):
        ax.bar(x + i * width, df[col], width, label=col, color=color, alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    print("[INFO] Model comparison saved")


def plot_feature_importance(models_dict, feature_names, output_dir="outputs"):
    """Plot feature importance for tree-based models (RF & XGBoost)."""
    for name, (model, _) in models_dict.items():
        if not hasattr(model, 'feature_importances_'):
            continue

        importances = model.feature_importances_
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        top_n = min(12, len(feature_names))
        top_idx = indices[:top_n]
        top_names = [feature_names[i] for i in top_idx]
        top_imp = importances[top_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.RdYlGn(top_imp / top_imp.max())
        ax.barh(range(top_n), top_imp[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance - {name}')
        plt.tight_layout()
        path = os.path.join(output_dir, f'feature_importance_{name.replace(" ", "_").lower()}.png')
        plt.savefig(path)
        plt.close()
        print(f"[INFO] Feature importance saved: {path}")
