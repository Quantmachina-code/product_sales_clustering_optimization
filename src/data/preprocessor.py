import pandas as pd
import numpy as np


def detect_violations(df):
    """
    Detect data quality violations:
    - price_elasticity > 0 (violates law of demand)
    - total_units_sold < 0 (negative quantities)
    - total_revenue < 0 (impossible transactions)

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with 'has_violation' column added
    """
    df = df.copy()

    df['has_violation'] = (
            (df['price_elasticity'] > 0) |
            (df['total_units_sold'] < 0) |
            (df['total_revenue'] < 0)
    )

    total_violations = df['has_violation'].sum()
    violation_pct = df['has_violation'].mean() * 100

    print(f"Total rows with violations: {total_violations} ({violation_pct:.2f}%)")

    # Check violations in labeled vs unlabeled
    if 'true_label' in df.columns:
        labeled_mask = df['true_label'].notna()
        labeled_violations = df[labeled_mask]['has_violation'].sum()
        labeled_total = labeled_mask.sum()

        unlabeled_violations = df[~labeled_mask]['has_violation'].sum()
        unlabeled_total = (~labeled_mask).sum()

        if labeled_total > 0:
            print(
                f"Labeled samples with violations: {labeled_violations}/{labeled_total} ({labeled_violations / labeled_total * 100:.2f}%)")
        if unlabeled_total > 0:
            print(
                f"Unlabeled samples with violations: {unlabeled_violations}/{unlabeled_total} ({unlabeled_violations / unlabeled_total * 100:.2f}%)")

    return df


def adjust_violations(df):
    """
    Adjust violations without removing rows (for predict mode):
    - price_elasticity > 0 → cap at 0
    - total_units_sold < 0 → set to 0
    - total_revenue < 0 → set to 0

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    df_adjusted : pd.DataFrame
        Dataframe with adjusted values
    """
    df_adjusted = df.copy()

    # Cap price_elasticity at 0
    if 'price_elasticity' in df_adjusted.columns:
        positive_elasticity = (df_adjusted['price_elasticity'] > 0).sum()
        if positive_elasticity > 0:
            df_adjusted.loc[df_adjusted['price_elasticity'] > 0, 'price_elasticity'] = 0
            print(f"Adjusted {positive_elasticity} positive price_elasticity values → 0")

    # Set negative units to 0
    if 'total_units_sold' in df_adjusted.columns:
        negative_units = (df_adjusted['total_units_sold'] < 0).sum()
        if negative_units > 0:
            df_adjusted.loc[df_adjusted['total_units_sold'] < 0, 'total_units_sold'] = 0
            print(f"Adjusted {negative_units} negative total_units_sold values → 0")

    # Set negative revenue to 0
    if 'total_revenue' in df_adjusted.columns:
        negative_revenue = (df_adjusted['total_revenue'] < 0).sum()
        if negative_revenue > 0:
            df_adjusted.loc[df_adjusted['total_revenue'] < 0, 'total_revenue'] = 0
            print(f"Adjusted {negative_revenue} negative total_revenue values → 0")

    return df_adjusted


def filter_violations(df):
    """
    Remove rows with violations (for train mode).

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'has_violation' column

    Returns:
    --------
    df_filtered : pd.DataFrame
        Dataframe with violations removed
    """
    df_filtered = df[~df['has_violation']].copy()

    # Drop the has_violation column
    if 'has_violation' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['has_violation'])

    removed = len(df) - len(df_filtered)
    print(f"Filtered out {removed} rows with violations")

    return df_filtered


def convert_units_to_int(df):
    """
    Convert total_units_sold to integer dtype.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with converted column
    """
    df = df.copy()

    if 'total_units_sold' in df.columns:
        df['total_units_sold'] = df['total_units_sold'].astype(int)
        print("Converted total_units_sold to integer dtype")

    return df


def check_duplicates(df):
    """
    Check for duplicate product_ids.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with duplicates removed (keeps first occurrence)
    """
    duplicates = df[df.duplicated(subset=['product_id'], keep=False)]
    n_duplicates = duplicates['product_id'].nunique()
    n_duplicate_rows = len(duplicates)

    print(f"\nDuplicate product_ids: {n_duplicates}")
    print(f"Total duplicate rows: {n_duplicate_rows}")

    if n_duplicate_rows > 0:
        # Remove duplicates, keep first
        df_clean = df.drop_duplicates(subset=['product_id'], keep='first').copy()
        print(f"Removed {n_duplicate_rows} duplicate rows, kept first occurrence")
        return df_clean
    else:
        print("No duplicate product_ids found.")
        return df.copy()


def preprocess_data(df, mode='train'):
    """
    Main preprocessing function following notebook logic.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    mode : str
        'train' (filter violations) or 'predict' (adjust violations)

    Returns:
    --------
    df_processed : pd.DataFrame
        Processed dataframe
    """
    print(f"\n=== Preprocessing ({mode} mode) ===")
    print(f"Initial shape: {df.shape}")

    df_processed = df.copy()

    # Detect violations
    df_processed = detect_violations(df_processed)

    if mode == 'train':
        # Filter out violations
        df_processed = filter_violations(df_processed)

        # Convert units to int
        df_processed = convert_units_to_int(df_processed)

        # Check and remove duplicates
        df_processed = check_duplicates(df_processed)

    elif mode == 'predict':
        # Adjust violations without removing rows
        df_processed = adjust_violations(df_processed)

        # Remove has_violation column if present
        if 'has_violation' in df_processed.columns:
            df_processed = df_processed.drop(columns=['has_violation'])

        # Convert units to int (after adjustment)
        df_processed = convert_units_to_int(df_processed)

    print(f"Final shape: {df_processed.shape}")
    print("=" * 50)

    return df_processed


if __name__ == "__main__":
    # Test preprocessing
    import io

    mock_csv = """product_id,price_elasticity,total_units_sold,total_revenue,total_online_views,true_label
1,0.5,100.5,500,200,KVI
2,-0.3,50.2,1000,300,SD
3,0.7,-10.0,750,400,
4,-0.8,200.3,-100,500,PG
5,-0.2,150.0,2000,600,"""

    df_test = pd.read_csv(io.StringIO(mock_csv))
    print("Original data:")
    print(df_test)
    print()

    # Test train mode (filter)
    print("\n" + "=" * 50)
    print("TRAIN MODE TEST")
    print("=" * 50)
    df_train = preprocess_data(df_test, mode='train')
    print("\nCleaned data (train mode):")
    print(df_train)
    print(f"Data types:\n{df_train.dtypes}")
    assert df_train['total_units_sold'].dtype == 'int64'
    assert len(df_train) < len(df_test)  # Some rows filtered
    assert (df_train['price_elasticity'] <= 0).all()
    assert (df_train['total_units_sold'] >= 0).all()
    assert (df_train['total_revenue'] >= 0).all()

    # Test predict mode (adjust)
    print("\n" + "=" * 50)
    print("PREDICT MODE TEST")
    print("=" * 50)
    df_predict = preprocess_data(df_test, mode='predict')
    print("\nAdjusted data (predict mode):")
    print(df_predict)
    assert len(df_predict) == len(df_test)  # No rows removed
    assert (df_predict['price_elasticity'] <= 0).all()
    assert (df_predict['total_units_sold'] >= 0).all()
    assert (df_predict['total_revenue'] >= 0).all()

    print("\n✓ preprocessor.py test passed")