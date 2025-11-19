import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import pickle


def fit_scaler(X, scaler_type='robust'):
    """
    Fit scaler on training data.

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Training features
    scaler_type : str
        Type of scaler ('robust' or 'standard')

    Returns:
    --------
    scaler : sklearn scaler object
        Fitted scaler
    """
    print(f"\n=== FITTING {scaler_type.upper()} SCALER ===\n")

    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

    scaler.fit(X)

    print(f"Scaler fitted on {X.shape[0]} samples, {X.shape[1]} features")

    return scaler


def transform_features(scaler, X, feature_names=None):
    """
    Transform features using fitted scaler.

    Parameters:
    -----------
    scaler : sklearn scaler object
        Fitted scaler
    X : pd.DataFrame or np.ndarray
        Features to transform
    feature_names : list, optional
        Column names for output DataFrame

    Returns:
    --------
    X_scaled : pd.DataFrame or np.ndarray
        Scaled features (returns DataFrame if feature_names provided)
    """
    X_scaled = scaler.transform(X)

    if feature_names is not None:
        X_scaled = pd.DataFrame(
            X_scaled,
            columns=feature_names,
            index=X.index if isinstance(X, pd.DataFrame) else None
        )

    print(f"Features transformed: {X_scaled.shape}")

    return X_scaled


def fit_transform_features(X, scaler_type='robust', feature_names=None):
    """
    Fit scaler and transform in one step.

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Training features
    scaler_type : str
        Type of scaler ('robust' or 'standard')
    feature_names : list, optional
        Column names for output DataFrame

    Returns:
    --------
    X_scaled : pd.DataFrame or np.ndarray
        Scaled features
    scaler : sklearn scaler object
        Fitted scaler
    """
    scaler = fit_scaler(X, scaler_type)
    X_scaled = transform_features(scaler, X, feature_names)

    print("\n=== SCALING COMPLETE ===\n")

    if isinstance(X_scaled, pd.DataFrame):
        print("Scaled data statistics (should be centered around 0):")
        print(X_scaled.describe().loc[['mean', '50%', 'std']].round(3))

    return X_scaled, scaler


if __name__ == "__main__":
    # Test scaler functionality
    import io

    mock_csv = """feature_1,feature_2,feature_3
    100,0.5,1000
    200,0.3,2000
    150,0.8,1500
    1000,0.2,5000
    50,0.9,500"""

    df_test = pd.read_csv(io.StringIO(mock_csv))
    print("Original data:")
    print(df_test)
    print("\nOriginal statistics:")
    print(df_test.describe())

    # Test fit_transform
    X_scaled, scaler = fit_transform_features(
        df_test,
        scaler_type='robust',
        feature_names=df_test.columns.tolist()
    )

    print("\nScaled data:")
    print(X_scaled)

    # Verify scaling properties
    assert isinstance(X_scaled, pd.DataFrame)
    assert X_scaled.shape == df_test.shape
    assert abs(X_scaled.median().mean()) < 1.0  # Should be near 0

    # Test transform on new data
    new_data = pd.DataFrame([[175, 0.6, 1750]], columns=df_test.columns)
    new_scaled = transform_features(scaler, new_data, feature_names=df_test.columns.tolist())

    print("\nNew data scaled:")
    print(new_scaled)

    print("\n✓ scaler.py test passed")