"""
Credit Card Fraud Detection - Preprocessing
Handles: ID removal, OneHot encoding, cyclic time encoding,
StandardScaler, SMOTE oversampling, train/test split.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess(df, use_smote=True, random_state=42):
    """Full preprocessing pipeline for the 10k fraud dataset.

    Steps:
    1. Drop 'transaction_id'
    2. OneHotEncode 'merchant_category'
    3. Cyclic encoding (sin/cos) for 'transaction_hour'
    4. StandardScaler on numeric columns
    5. SMOTE oversampling (optional)
    6. Train/test split (80/20, stratified)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    use_smote : bool
        Whether to apply SMOTE oversampling.
    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Preprocessed data splits.
    feature_names : list
        Names of features after preprocessing (for interpretability).
    """
    data = df.copy()

    # Step 1: Drop ID column
    data.drop('transaction_id', axis=1, inplace=True)
    print("[INFO] Dropped 'transaction_id'")

    # Step 2: OneHotEncode merchant_category
    ohe = OneHotEncoder(sparse=False, dtype=np.float64)
    cat_encoded = ohe.fit_transform(data[['merchant_category']])
    cat_names = ohe.get_feature_names(['merchant_category']).tolist()
    cat_df = pd.DataFrame(cat_encoded, columns=cat_names, index=data.index)
    data = pd.concat([data.drop('merchant_category', axis=1), cat_df], axis=1)
    print(f"[INFO] OneHotEncoded 'merchant_category' -> {len(cat_names)} columns: {cat_names}")

    # Step 3: Cyclic encoding for transaction_hour
    hours = data['transaction_hour'].values
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    data['hour_sin'] = hour_sin
    data['hour_cos'] = hour_cos
    data.drop('transaction_hour', axis=1, inplace=True)
    print("[INFO] Cyclic encoded 'transaction_hour' -> 'hour_sin', 'hour_cos'")

    # Step 4: Scale all numeric features
    feature_cols = [c for c in data.columns if c != 'is_fraud']
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    print(f"[INFO] StandardScaler applied to {len(feature_cols)} features")

    # Split features and target
    X = data.drop('is_fraud', axis=1).values
    y = data['is_fraud'].values
    final_feature_names = data.drop('is_fraud', axis=1).columns.tolist()

    # Step 5: SMOTE
    if use_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X, y = smote.fit_resample(X, y)
        print(f"[INFO] SMOTE applied: {X.shape[0]} samples ({np.sum(y==0)} normal, {np.sum(y==1)} fraud)")

    # Step 6: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train set: {X_train.shape[0]} samples, fraud: {np.sum(y_train)} ({np.sum(y_train)/len(y_train)*100:.2f}%)")
    print(f"[INFO] Test set:  {X_test.shape[0]} samples, fraud: {np.sum(y_test)} ({np.sum(y_test)/len(y_test)*100:.2f}%)")

    return X_train, X_test, y_train, y_test, final_feature_names
