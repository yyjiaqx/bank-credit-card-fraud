"""
Credit Card Fraud Detection - Data Loader
Loads and inspects the credit card transaction dataset (10k version).
"""
import pandas as pd
import numpy as np
import os


def load_data(data_path="data/credit_card_fraud_10k.csv"):
    """Load the credit card fraud dataset.

    Parameters
    ----------
    data_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    if not os.path.exists(data_path):
        kaggle_path = "/kaggle/input/creditcardfraud/credit_card_fraud_10k.csv"
        if os.path.exists(kaggle_path):
            data_path = kaggle_path
            print("[INFO] Using Kaggle dataset path")
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{data_path}'. "
                "Please place 'credit_card_fraud_10k.csv' in the 'data/' folder."
            )
    df = pd.read_csv(data_path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def basic_info(df):
    """Print basic information about the dataset."""
    print("=" * 50)
    print("Dataset Overview")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData Types:")
    print(df.dtypes.value_counts().to_string())
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  No missing values found.")
    else:
        print(missing.to_string())


def class_distribution(df):
    """Print and return the class distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing an 'is_fraud' column.

    Returns
    -------
    pd.Series
        Class counts.
    """
    counts = df['is_fraud'].value_counts()
    fraud_pct = counts.get(1, 0) / df.shape[0] * 100

    print("=" * 50)
    print("Class Distribution")
    print("=" * 50)
    print(f"Normal transactions: {counts.get(0, 0):,}")
    print(f"Fraud transactions:  {counts.get(1, 0):,}")
    print(f"Fraud percentage:    {fraud_pct:.2f}%")
    return counts


def describe_features(df):
    """Print statistical summary."""
    print("=" * 50)
    print("Statistical Summary")
    print("=" * 50)
    print(df.describe().round(2).to_string())


def feature_correlations(df):
    """Print feature correlations with the fraud target."""
    print("=" * 50)
    print("Correlation with is_fraud")
    print("=" * 50)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in sorted(numeric_cols):
        if col != 'is_fraud':
            corr = df[col].corr(df['is_fraud'])
            bar = '#' * int(abs(corr) * 50)
            print(f"  {col:25s}: {corr:+.4f}  {bar}")


def categorical_breakdown(df):
    """Print category distribution and fraud rates by category."""
    print("=" * 50)
    print("Fraud Rate by Merchant Category")
    print("=" * 50)
    for cat in sorted(df['merchant_category'].unique()):
        subset = df[df['merchant_category'] == cat]
        fraud_rate = subset['is_fraud'].mean() * 100
        print(f"  {cat:15s}: {len(subset):5d} txns, fraud={fraud_rate:.2f}%")
