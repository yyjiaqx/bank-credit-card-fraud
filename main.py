"""
Credit Card Fraud Detection - Main Script (10k dataset)
Run the full pipeline: load -> EDA -> preprocess -> train -> evaluate -> SHAP.

Usage:
    python main.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, basic_info, class_distribution, categorical_breakdown
from src.preprocessing import preprocess
from src.models import train_all_models, shap_analysis
from src.evaluation import (
    evaluate_model, plot_confusion_matrix, plot_roc_curves,
    plot_pr_curves, plot_model_comparison, plot_feature_importance
)
from sklearn.metrics import confusion_matrix


def main():
    if os.path.exists("/kaggle/input"):
        DATA_PATH = "/kaggle/input/creditcardfraud/credit_card_fraud_10k.csv"
        OUTPUT_DIR = "/kaggle/working"
        print("[INFO] Running on Kaggle")
    else:
        DATA_PATH = "data/credit_card_fraud_10k.csv"
        OUTPUT_DIR = "outputs"
        print("[INFO] Running locally")

    RANDOM_STATE = 42
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load + EDA
    print("\n" + "#" * 60)
    print("#  STEP 1: Data Loading & EDA")
    print("#" * 60)
    df = load_data(DATA_PATH)
    basic_info(df)
    class_distribution(df)
    categorical_breakdown(df)

    # Step 2: Preprocess
    print("\n" + "#" * 60)
    print("#  STEP 2: Preprocessing")
    print("#" * 60)
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df, use_smote=True, random_state=RANDOM_STATE
    )
    print(f"\n[INFO] Final feature set ({len(feature_names)}):")
    for i, name in enumerate(feature_names):
        print(f"  {i:2d}. {name}")

    # Step 3: Train
    print("\n" + "#" * 60)
    print("#  STEP 3: Model Training")
    print("#" * 60)
    models_dict = train_all_models(X_train, y_train, RANDOM_STATE)

    # Step 4: Evaluate
    print("\n" + "#" * 60)
    print("#  STEP 4: Model Evaluation")
    print("#" * 60)
    metrics_list = []
    for name, (model, params) in models_dict.items():
        metrics = evaluate_model(model, name, X_test, y_test, OUTPUT_DIR)
        cm = confusion_matrix(y_test, model.predict(X_test))
        plot_confusion_matrix(cm, name, OUTPUT_DIR)
        metrics['Best Params'] = str(params)
        metrics_list.append(metrics)

    # Step 5: Comparison
    print("\n" + "#" * 60)
    print("#  STEP 5: Model Comparison")
    print("#" * 60)
    plot_roc_curves(models_dict, X_test, y_test, OUTPUT_DIR)
    plot_pr_curves(models_dict, X_test, y_test, OUTPUT_DIR)
    plot_model_comparison(metrics_list, OUTPUT_DIR)
    plot_feature_importance(models_dict, feature_names, OUTPUT_DIR)

    # Step 6: SHAP Explainability
    print("\n" + "#" * 60)
    print("#  STEP 6: SHAP Explainability")
    print("#" * 60)

    import pandas as pd
    summary = pd.DataFrame(metrics_list).set_index('Model')
    best_model_name = summary['ROC-AUC'].idxmax()

    shap_analysis(models_dict['Logistic Regression'][0], X_train, X_test,
                  feature_names, 'Logistic Regression', OUTPUT_DIR)
    shap_analysis(models_dict[best_model_name][0], X_train, X_test,
                  feature_names, best_model_name, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(summary[['ROC-AUC', 'PR-AUC', 'Recall', 'Precision', 'F1-Score']].round(4).to_string())
    print(f"\n  Best Model: {best_model_name}")

    print("\n" + "=" * 60)
    print("  Pipeline complete! Check 'outputs/' for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
