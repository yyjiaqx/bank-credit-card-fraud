"""
Credit Card Fraud Detection - Models
Defines and trains Logistic Regression, Random Forest, and XGBoost classifiers
with GridSearchCV hyperparameter tuning.
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
