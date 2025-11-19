import sys
import os
import pandas as pd
import numpy as np
import random
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_data, get_labeled_data, get_unlabeled_data
from src.data.preprocessor import preprocess_data
from src.features.engineer import engineer_features
from src.features.outliers import detect_multivariate_outliers
from src.features.scaler import fit_transform_features
from src.models.decision_tree import train_decision_tree
from src.models.clustering import train_hierarchical_clustering
from src.models.gmm import train_gmm
from src.utils.io_utils import save_model

# Configuration
RANDOM_STATE = 42

# Seed for reproducibility (must be set before any random operations)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore')


def main():
    """
    Main training pipeline.

    Steps:
    1. Load raw data
    2. Preprocess (filter violations, convert types, remove duplicates)
    3. Engineer features
    4. Detect and remove multivariate outliers
    5. Fit RobustScaler on clean data
    6. Train Decision Tree on labeled data
    7. Extract top 4 features from Decision Tree
    8. Train Hierarchical Clustering on unlabeled data (top 4 features)
    9. Train GMM on unlabeled data (top 4 features)
    10. Save all models and scaler
    """

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE")
    print("=" * 80 + "\n")

    # Configuration - resolve paths relative to script location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent of scripts/

    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "data.parquet")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

    print(f"Random seed: {RANDOM_STATE}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data path: {DATA_PATH}")
    print(f"Models directory: {MODELS_DIR}")

    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ========================================================================
    # STEP 1: Load raw data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)

    df_raw = load_data(DATA_PATH)
    print(f"Columns: {df_raw.columns.tolist()}")

    # ========================================================================
    # STEP 2: Preprocess data (train mode - filter violations)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: PREPROCESS DATA")
    print("=" * 80)

    df_clean = preprocess_data(df_raw, mode='train')

    # ========================================================================
    # STEP 3: Feature engineering
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 80)

    df_engineered, engineered_feature_names = engineer_features(df_clean)

    # Get all numeric features (original + engineered)
    numeric_features = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
    if 'product_id' in numeric_features:
        numeric_features.remove('product_id')

    print(f"\nTotal numeric features: {len(numeric_features)}")

    # ========================================================================
    # STEP 4: Remove multivariate outliers
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: REMOVE MULTIVARIATE OUTLIERS")
    print("=" * 80)

    df_no_outliers, outlier_mask = detect_multivariate_outliers(
        df_engineered,
        feature_columns=numeric_features,
        contamination=0.05,
        random_state=RANDOM_STATE
    )

    # ========================================================================
    # STEP 5: Fit RobustScaler
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: FIT ROBUST SCALER")
    print("=" * 80)

    # Prepare feature matrix (exclude product_id and true_label)
    X = df_no_outliers[numeric_features].copy()

    # Fit and transform
    X_scaled, scaler = fit_transform_features(
        X,
        scaler_type='robust',
        feature_names=numeric_features
    )

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "robust_scaler.pkl")
    save_model(scaler, scaler_path)

    # Add metadata back
    X_scaled['product_id'] = df_no_outliers['product_id'].values
    X_scaled['true_label'] = df_no_outliers['true_label'].values

    # ========================================================================
    # STEP 6: Separate labeled and unlabeled data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: SEPARATE LABELED AND UNLABELED DATA")
    print("=" * 80)

    labeled_mask = X_scaled['true_label'].notna()
    unlabeled_mask = ~labeled_mask

    X_labeled = X_scaled[labeled_mask][numeric_features].copy()
    y_labeled = X_scaled[labeled_mask]['true_label'].copy()

    X_unlabeled = X_scaled[unlabeled_mask][numeric_features].copy()

    # Keep reference to full engineered dataframe for clustering mapping
    df_unlabeled_full = df_no_outliers[unlabeled_mask].copy()

    print(f"Labeled samples: {len(X_labeled)}")
    print(f"Unlabeled samples: {len(X_unlabeled)}")
    print(f"\nLabel distribution:")
    print(y_labeled.value_counts())

    # ========================================================================
    # STEP 7: Train Decision Tree on labeled data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: TRAIN DECISION TREE")
    print("=" * 80)

    dt_model, dt_importances, dt_top_features = train_decision_tree(
        X_labeled,
        y_labeled,
        feature_names=numeric_features,
        max_depth=3,
        max_features=4,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )

    # Save Decision Tree
    dt_path = os.path.join(MODELS_DIR, "decision_tree.pkl")
    save_model(dt_model, dt_path)

    # ========================================================================
    # EXTRACT TOP 4 FEATURES FROM DECISION TREE
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXTRACTING TOP 4 FEATURES FOR CLUSTERING")
    print("=" * 80 + "\n")

    top_4_features = dt_top_features[:4]
    print(f"Top 4 features selected for clustering:")
    for i, feat in enumerate(top_4_features, 1):
        importance = dt_importances[dt_importances['feature'] == feat]['importance'].values[0]
        print(f"  {i}. {feat} (importance: {importance:.4f})")

    # Subset data to top 4 features for clustering
    X_unlabeled_top4 = X_unlabeled[top_4_features].copy()

    print(f"\nUnlabeled data shape for clustering: {X_unlabeled_top4.shape}")

    # ========================================================================
    # STEP 8: Train Hierarchical Clustering on unlabeled data (top 4 features)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: TRAIN HIERARCHICAL CLUSTERING (TOP 4 FEATURES)")
    print("=" * 80)

    # Best parameters from notebook: linkage='ward', metric='euclidean'
    hierarchical_model, knn_predictor, hierarchical_mapping, hierarchical_labels = train_hierarchical_clustering(
        X_unlabeled_top4,
        df_unlabeled_full,
        linkage='ward',
        metric='euclidean',
        top_features=top_4_features,
        random_state=RANDOM_STATE
    )

    # Save Hierarchical Clustering (save the KNN predictor since hierarchical has no predict)
    hierarchical_bundle = {
        'model': hierarchical_model,
        'knn_predictor': knn_predictor,
        'cluster_mapping': hierarchical_mapping,
        'top_features': top_4_features,
        'linkage': 'ward',
        'metric': 'euclidean'
    }
    hierarchical_path = os.path.join(MODELS_DIR, "monotonic_clustering.pkl")
    save_model(hierarchical_bundle, hierarchical_path)

    # ========================================================================
    # STEP 9: Train GMM on unlabeled data (top 4 features)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: TRAIN GAUSSIAN MIXTURE MODEL (TOP 4 FEATURES)")
    print("=" * 80)

    # Best hyperparameters from notebook: covariance_type='full', n_init=10, max_iter=200
    gmm_model, gmm_mapping, gmm_labels = train_gmm(
        X_unlabeled_top4,
        df_unlabeled_full,
        covariance_type='full',
        n_init=10,
        max_iter=200,
        top_features=top_4_features,
        random_state=RANDOM_STATE
    )

    # Save GMM
    gmm_bundle = {
        'model': gmm_model,
        'cluster_mapping': gmm_mapping,
        'top_features': top_4_features,
        'covariance_type': 'full',
        'n_init': 10,
        'max_iter': 200
    }
    gmm_path = os.path.join(MODELS_DIR, "gmm_model.pkl")
    save_model(gmm_bundle, gmm_path)

    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80 + "\n")

    print("Models saved:")
    print(f"  - {scaler_path}")
    print(f"  - {dt_path}")
    print(f"  - {hierarchical_path}")
    print(f"  - {gmm_path}")

    print("\nMetadata:")
    print(f"  Random seed: {RANDOM_STATE}")
    print(f"  Total features: {len(numeric_features)}")
    print(f"  Top 4 features for clustering: {top_4_features}")
    print(f"  Labeled samples: {len(X_labeled)}")
    print(f"  Unlabeled samples: {len(X_unlabeled)}")
    print(f"  Total samples after cleaning: {len(df_no_outliers)}")

    # Save feature names for reference
    feature_metadata = {
        'numeric_features': numeric_features,
        'engineered_features': engineered_feature_names,
        'top_4_features': top_4_features,
        'random_state': RANDOM_STATE
    }
    metadata_path = os.path.join(MODELS_DIR, "feature_metadata.pkl")
    save_model(feature_metadata, metadata_path)
    print(f"  - {metadata_path}")


if __name__ == "__main__":
    main()