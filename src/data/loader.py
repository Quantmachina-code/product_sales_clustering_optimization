import pandas as pd
from pathlib import Path


def load_data(filepath):
    """
    Load dataset from CSV or Parquet file.

    Parameters:
    -----------
    filepath : str
        Path to CSV or Parquet file

    Returns:
    --------
    df : pd.DataFrame
        Loaded dataframe
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Determine file type by extension
    file_ext = Path(filepath).suffix.lower()

    if file_ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif file_ext == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .csv or .parquet")

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def get_labeled_data(df):
    """
    Extract rows with non-null true_label.

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset

    Returns:
    --------
    labeled_df : pd.DataFrame
        Subset with true_label present
    """
    labeled_df = df[df['true_label'].notna()].copy()
    print(f"Labeled samples: {len(labeled_df)}")
    return labeled_df


def get_unlabeled_data(df):
    """
    Extract rows with null true_label.

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset

    Returns:
    --------
    unlabeled_df : pd.DataFrame
        Subset without true_label
    """
    unlabeled_df = df[df['true_label'].isna()].copy()
    print(f"Unlabeled samples: {len(unlabeled_df)}")
    return unlabeled_df


if __name__ == "__main__":
    # Test with mock data
    print("Testing loader.py with mock data...")

    import io

    mock_csv = """product_id,price_elasticity,total_online_views,total_units_sold,total_revenue,true_label
1,0.5,100,50,500,KVI
2,0.3,200,100,1000,
3,0.7,150,75,750,SD"""

    df_test = pd.read_csv(io.StringIO(mock_csv))

    print(f"\nTest data shape: {df_test.shape}")

    labeled = get_labeled_data(df_test)
    assert len(labeled) == 2

    unlabeled = get_unlabeled_data(df_test)
    assert len(unlabeled) == 1

    print("✓ loader.py test passed")