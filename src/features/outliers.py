import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_multivariate_outliers(df, feature_columns, contamination=0.05, random_state=42):
    """
    Detect multivariate outliers using Isolation Forest.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with engineered features
    feature_columns : list
        List of feature column names to use for outlier detection
    contamination : float
        Expected proportion of outliers (default: 0.05 = 5%)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    df_clean : pd.DataFrame
        Dataframe with outliers removed
    outlier_mask : pd.Series
        Boolean mask indicating outliers (True = outlier)
    """
    print("\n=== MULTIVARIATE OUTLIER DETECTION ===\n")

    # Prepare numeric data
    df_numeric = df[feature_columns].copy()

    # Scale features for Isolation Forest
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

    # Predict outliers (-1 for outliers, 1 for inliers)
    outlier_labels = iso_forest.fit_predict(df_scaled)
    outlier_mask = (outlier_labels == -1)

    n_outliers = outlier_mask.sum()
    outlier_pct = outlier_mask.mean() * 100

    print(f"Multivariate outliers detected: {n_outliers} ({outlier_pct:.2f}%)")

    # Remove outliers
    df_clean = df[~outlier_mask].copy()

    print(f"Remaining samples: {len(df_clean)}")

    return df_clean, outlier_mask


if __name__ == "__main__":
    # Test multivariate outlier detection
    import io
    from sklearn.datasets import make_classification

    # Create synthetic data with outliers
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )

    # Add extreme outliers
    X[:5] = X[:5] * 10

    df_test = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df_test['product_id'] = range(len(df_test))
    df_test['true_label'] = ['KVI'] * 67 + ['SD'] * 66 + ['PG'] * 67

    print("Test data shape:", df_test.shape)

    feature_cols = [f'feature_{i}' for i in range(10)]

    # Test outlier detection
    df_clean, outlier_mask = detect_multivariate_outliers(
        df_test,
        feature_cols,
        contamination=0.05,
        random_state=42
    )

    print("\nCleaned data shape:", df_clean.shape)
    assert len(df_clean) < len(df_test)
    assert outlier_mask.sum() > 0

    print("\n✓ outliers.py test passed")