import pandas as pd
import numpy as np

# Safety epsilon for division
EPSILON = 1e-6


def create_ratio_features(df):
    """
    Create ratio-based features for margin and efficiency.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with base features

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with ratio features added
    """
    df = df.copy()

    df['revenue_per_unit'] = (
            df['total_revenue'] / (df['total_units_sold'] + EPSILON)
    )

    df['conversion_rate'] = (
            df['total_units_sold'] / (df['total_online_views'] + EPSILON)
    )

    df['basket_value_ratio'] = (
            df['mean_added_revenue_to_basket'] / (df['total_revenue'] + EPSILON)
    )

    df['profit_margin_proxy'] = (
            df['mean_basket_profit'] / (df['mean_added_revenue_to_basket'] + EPSILON)
    )

    print("  ✓ revenue_per_unit")
    print("  ✓ conversion_rate")
    print("  ✓ basket_value_ratio")
    print("  ✓ profit_margin_proxy")

    return df


def create_competitive_features(df):
    """
    Create competitive pressure metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with base features

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with competitive features added
    """
    df = df.copy()

    df['competitive_pressure'] = (
            df['competitive_price_range'] / (df['revenue_per_unit'] + EPSILON)
    )

    df['price_stability'] = (
            1 / (df['price_volatility'] + 0.001)
    )

    print("  ✓ competitive_pressure")
    print("  ✓ price_stability")

    return df


def create_interaction_features(df):
    """
    Create behavioral interaction terms.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with base features

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with interaction features added
    """
    df = df.copy()

    df['sales_velocity'] = (
            (df['total_units_sold'] / (df['total_online_views'] + EPSILON)) *
            df['median_basket_position']
    )

    df['elasticity_revenue_interaction'] = (
            df['price_elasticity'] * df['total_revenue']
    )

    df['volatility_competition_interaction'] = (
            df['price_volatility'] * df['competitive_price_range']
    )

    print("  ✓ sales_velocity")
    print("  ✓ elasticity_revenue_interaction")
    print("  ✓ volatility_competition_interaction")

    return df


def create_positional_features(df):
    """
    Create positional features based on basket behavior.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with base features

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with positional features added
    """
    df = df.copy()

    df['early_basket_indicator'] = (
            df['median_basket_position'] <= 2
    ).astype(int)

    df['basket_profit_lift'] = (
            df['mean_basket_profit'] -
            (df['total_revenue'] / (df['total_units_sold'] + EPSILON))
    )

    print("  ✓ early_basket_indicator")
    print("  ✓ basket_profit_lift")

    return df


def validate_engineered_features(df, engineered_features):
    """
    Check for inf/nan values and impute if necessary.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with engineered features
    engineered_features : list
        List of engineered feature names

    Returns:
    --------
    df : pd.DataFrame
        Validated dataframe
    """
    df = df.copy()

    print("\n=== ENGINEERED FEATURE QUALITY CHECK ===\n")

    issues_found = False
    for feat in engineered_features:
        n_inf = np.isinf(df[feat]).sum()
        n_nan = df[feat].isna().sum()

        if n_inf > 0 or n_nan > 0:
            issues_found = True
            print(f"{feat}:")
            if n_inf > 0:
                print(f"  Inf values: {n_inf}")
            if n_nan > 0:
                print(f"  NaN values: {n_nan}")

    if not issues_found:
        print("No issues detected in engineered features.")

    # Handle any remaining inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaN with median
    for feat in engineered_features:
        if df[feat].isna().sum() > 0:
            median_value = df[feat].median()
            df[feat].fillna(median_value, inplace=True)
            print(f"Imputed {feat} with median ({median_value:.4f})")

    print("\nAll engineered features validated.")

    return df


def engineer_features(df):
    """
    Main feature engineering function.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe

    Returns:
    --------
    df_engineered : pd.DataFrame
        Dataframe with all engineered features
    feature_names : list
        List of all engineered feature names
    """
    print("\n=== FEATURE ENGINEERING ===\n")
    print("Creating derived features...\n")

    df_engineered = df.copy()

    # 1. Ratio-based features
    print("1. Ratio-based features")
    df_engineered = create_ratio_features(df_engineered)

    # 2. Competitive pressure metrics
    print("\n2. Competitive pressure metrics")
    df_engineered = create_competitive_features(df_engineered)

    # 3. Behavioral interaction terms
    print("\n3. Behavioral interaction terms")
    df_engineered = create_interaction_features(df_engineered)

    # 4. Positional features
    print("\n4. Positional features")
    df_engineered = create_positional_features(df_engineered)

    # List of all engineered features
    engineered_features = [
        'revenue_per_unit', 'conversion_rate', 'basket_value_ratio', 'profit_margin_proxy',
        'competitive_pressure', 'price_stability', 'sales_velocity',
        'elasticity_revenue_interaction', 'volatility_competition_interaction',
        'early_basket_indicator', 'basket_profit_lift'
    ]

    # Validate engineered features
    df_engineered = validate_engineered_features(df_engineered, engineered_features)

    # Get original numeric features (exclude product_id and true_label)
    original_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'product_id' in original_features:
        original_features.remove('product_id')

    all_features = original_features + engineered_features

    print(f"\n=== FEATURE ENGINEERING COMPLETE ===")
    print(f"Original features: {len(original_features)}")
    print(f"Engineered features: {len(engineered_features)}")
    print(f"Total features: {len(all_features)}")

    return df_engineered, engineered_features


if __name__ == "__main__":
    # Test feature engineering
    import io

    mock_csv = """product_id,price_elasticity,total_online_views,total_units_sold,total_revenue,median_basket_position,mean_added_revenue_to_basket,mean_basket_profit,competitive_price_range,price_volatility,true_label
1,-0.5,200,100,500,3,50,100,10,0.1,KVI
2,-0.3,300,150,1000,5,75,200,15,0.2,SD
3,-0.8,400,200,2000,1,100,400,5,0.05,PG"""

    df_test = pd.read_csv(io.StringIO(mock_csv))
    print("Original data:")
    print(df_test)
    print()

    # Test feature engineering
    df_engineered, engineered_features = engineer_features(df_test)

    print("\n=== ENGINEERED DATA SAMPLE ===")
    sample_cols = ['product_id', 'true_label'] + engineered_features
    print(df_engineered[sample_cols])

    # Verify features exist
    for feat in engineered_features:
        assert feat in df_engineered.columns, f"Missing feature: {feat}"

    # Verify no inf/nan
    for feat in engineered_features:
        assert not np.isinf(df_engineered[feat]).any(), f"Inf in {feat}"
        assert not df_engineered[feat].isna().any(), f"NaN in {feat}"

    print("\n✓ engineer.py test passed")