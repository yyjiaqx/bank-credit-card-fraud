"""
Credit Card Fraud Detection - Models
Defines and trains Logistic Regression, Random Forest, and XGBoost classifiers
with GridSearchCV hyperparameter tuning and SHAP explainability analysis.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb


def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression with GridSearchCV."""
    print("\n" + "=" * 50)
    print("Training: Logistic Regression")
    print("=" * 50)

    params = {'C': [0.1, 1.0], 'max_iter': [2000]}
    lr = LogisticRegression(random_state=random_state, max_iter=2000)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    grid = GridSearchCV(lr, params, cv=cv, scoring='recall', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV recall: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def train_random_forest(X_train, y_train, random_state=42):
    """Train Random Forest with GridSearchCV."""
    print("\n" + "=" * 50)
    print("Training: Random Forest")
    print("=" * 50)

    params = {
        'n_estimators': [100],
        'max_depth': [8, 12],
        'min_samples_split': [5],
    }
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    grid = GridSearchCV(rf, params, cv=cv, scoring='recall', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV recall: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def train_xgboost(X_train, y_train, random_state=42):
    """Train XGBoost with GridSearchCV."""
    print("\n" + "=" * 50)
    print("Training: XGBoost")
    print("=" * 50)

    params = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.1],
    }
    xgb_model = xgb.XGBClassifier(
        random_state=random_state, n_jobs=-1,
        use_label_encoder=False, eval_metric='logloss'
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    grid = GridSearchCV(xgb_model, params, cv=cv, scoring='recall', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV recall: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def train_all_models(X_train, y_train, random_state=42):
    """Train all three models and return them."""
    results = {}

    lr_model, lr_params = train_logistic_regression(X_train, y_train, random_state)
    results['Logistic Regression'] = (lr_model, lr_params)

    rf_model, rf_params = train_random_forest(X_train, y_train, random_state)
    results['Random Forest'] = (rf_model, rf_params)

    xgb_model, xgb_params = train_xgboost(X_train, y_train, random_state)
    results['XGBoost'] = (xgb_model, xgb_params)

    return results

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def shap_analysis(model, X_train, X_test, feature_names, model_name,
                  output_dir="outputs", max_display=20):
    """SHAP explainability: Summary Plot, Waterfall, Dependence Plot.

    Parameters
    ----------
    model : trained classifier
    X_train : np.ndarray — used to build explainer
    X_test : np.ndarray — used for prediction explanation
    feature_names : list of str
    model_name : str
    output_dir : str
    max_display : int
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        import shap
    except ImportError:
        print(f"  [WARN] shap not installed, skip SHAP. pip install shap")
        return

    print(f"\n  SHAP Analysis — {model_name}")
    print("  " + "-" * 50)

    sample_size = min(500, X_train.shape[0])
    X_train_sample = X_train[:sample_size]
    X_test_sample = X_test[:min(500, X_test.shape[0])]

    if hasattr(model, 'predict_proba') and hasattr(model, 'coef_'):
        explainer = shap.LinearExplainer(model, X_train_sample)
        shap_values = explainer.shap_values(X_test_sample)
    elif hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
    else:
        print(f"  [WARN] {model_name} does not support SHAP")
        return

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    safe_name = model_name.lower().replace(' ', '_')

    # Summary Plot
    print("  Generating Summary Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    path = os.path.join(output_dir, f'shap_summary_{safe_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")

    # Bar Plot
    print("  Generating Summary Bar Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                      plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    path = os.path.join(output_dir, f'shap_bar_{safe_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")

    # Waterfall Plot
    print("  Generating Waterfall Plot...")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        base_val = explainer.expected_value
        if isinstance(base_val, list):
            base_val = base_val[1]
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[0],
                             base_values=base_val,
                             data=X_test_sample[0],
                             feature_names=feature_names),
            max_display=15, show=False
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f'shap_waterfall_{safe_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] {path}")
    except Exception as e:
        print(f"  [Hint] Waterfall plot skipped: {e}")

    # Dependence Plot
    print("  Generating Dependence Plot...")
    try:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = int(np.argmax(mean_abs_shap))
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(top_idx, shap_values, X_test_sample,
                             feature_names=feature_names, show=False)
        plt.tight_layout()
        path = os.path.join(output_dir, f'shap_dependence_{safe_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Saved] {path}")
    except Exception as e:
        print(f"  [Hint] Dependence plot skipped: {e}")

    print(f"  SHAP Analysis complete — {model_name}")
