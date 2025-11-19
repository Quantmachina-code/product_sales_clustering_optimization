import sys
import os
import pandas as pd
import numpy as np
import random
import warnings
from scipy import stats

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.features.engineer import engineer_features
from src.features.scaler import transform_features
from src.utils.io_utils import load_model

# Configuration
RANDOM_STATE = 42

# Seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore')



def adjust_predictions_with_cluster_distances(df, X_scaled, clustering_model, cluster_mapping,
                                              initial_predictions, target_distribution,
                                              top_features, model_type='gmm'):
    """
    Adjust predictions to match target distribution using cluster distances.

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataframe with engineered features
    X_scaled : np.ndarray or pd.DataFrame
        Scaled features (top 4 features used by clustering)
    clustering_model : sklearn model
        Trained clustering model (GMM or Hierarchical)
    cluster_mapping : dict
        Mapping from cluster ID to segment label
    initial_predictions : np.ndarray
        Initial ensemble predictions
    target_distribution : dict
        Target distribution {'KVI': 0.10, 'SD': 0.30, 'PG': 0.60}
    top_features : list
        List of top 4 feature names used for clustering
    model_type : str
        'gmm' or 'hierarchical'

    Returns:
    --------
    adjusted_predictions : np.ndarray
        Predictions adjusted to match target distribution
    reallocation_info : dict
        Information about reallocation process
    """

    print("\n" + "=" * 80)
    print("APPLYING DISTRIBUTION CONSTRAINTS WITH CLUSTER DISTANCES")
    print("=" * 80 + "\n")

    n_samples = len(df)
    target_counts = {seg: int(n_samples * prop) for seg, prop in target_distribution.items()}

    # Adjust for rounding
    diff = n_samples - sum(target_counts.values())
    if diff != 0:
        max_seg = max(target_distribution, key=target_distribution.get)
        target_counts[max_seg] += diff

    print(f"Target distribution ({n_samples} samples):")
    for seg in ['KVI', 'SD', 'PG']:
        count = target_counts[seg]
        print(f"  {seg}: {count:5d} ({count / n_samples * 100:5.1f}%)")
    print()

    # Current distribution
    current_counts = pd.Series(initial_predictions).value_counts().to_dict()
    for seg in ['KVI', 'SD', 'PG']:
        if seg not in current_counts:
            current_counts[seg] = 0

    print("Current distribution:")
    for seg in ['KVI', 'SD', 'PG']:
        count = current_counts[seg]
        print(f"  {seg}: {count:5d} ({count / n_samples * 100:5.1f}%)")
    print()

    # Get cluster centers and labels based on model type
    if model_type == 'gmm':
        cluster_centers = clustering_model.means_
        cluster_labels = clustering_model.predict(X_scaled)
    else:  # hierarchical - use KNN predictor
        cluster_labels = clustering_model.predict(X_scaled)
        # Calculate centroids from assignments
        cluster_centers = np.array([X_scaled[cluster_labels == i].mean(axis=0) for i in range(3)])

    # Create reverse mapping: segment -> cluster_id
    reverse_mapping = {v: k for k, v in cluster_mapping.items()}

    # Calculate distances from each point to all cluster centers
    df_work = pd.DataFrame()
    df_work['initial_prediction'] = initial_predictions
    df_work['cluster_label'] = cluster_labels

    for seg in ['KVI', 'SD', 'PG']:
        cluster_id = reverse_mapping[seg]
        center = cluster_centers[cluster_id]
        # Euclidean distance to cluster center
        distances = np.linalg.norm(X_scaled - center, axis=1)
        df_work[f'dist_to_{seg}'] = distances

    # Calculate surplus and deficit
    surplus = {}
    deficit = {}
    for seg in ['KVI', 'SD', 'PG']:
        diff = current_counts[seg] - target_counts[seg]
        if diff > 0:
            surplus[seg] = diff
        elif diff < 0:
            deficit[seg] = -diff

    print("Adjustments needed:")
    if surplus:
        print("  Surplus (need to remove):")
        for seg, count in surplus.items():
            print(f"    {seg}: {count}")
    if deficit:
        print("  Deficit (need to add):")
        for seg, count in deficit.items():
            print(f"    {seg}: {count}")
    print()

    # Initialize adjusted predictions
    adjusted_predictions = df_work['initial_prediction'].copy().values
    reallocation_log = []

    # Reallocate from surplus to deficit segments
    for surplus_seg, surplus_count in surplus.items():
        surplus_mask = (adjusted_predictions == surplus_seg)
        surplus_indices = np.where(surplus_mask)[0]

        if len(surplus_indices) == 0:
            continue

        candidates = df_work.iloc[surplus_indices].copy()

        # Calculate reallocation score: weighted distance to deficit segments
        # Lower distance = higher score (more suitable for reallocation)
        candidates['reallocation_score'] = 0
        for deficit_seg in deficit.keys():
            # Use inverse distance as score (closer = higher score)
            candidates['reallocation_score'] += (1.0 / (candidates[f'dist_to_{deficit_seg}'] + 0.01))

        # Also consider distance from current cluster (further = easier to reallocate)
        candidates['reallocation_score'] += candidates[f'dist_to_{surplus_seg}'] * 0.5

        # Sort by reallocation score (higher = better candidate)
        candidates_sorted = candidates.sort_values('reallocation_score', ascending=False)

        # Reallocate samples
        for idx in candidates_sorted.index[:surplus_count]:
            old_seg = surplus_seg

            # Find best deficit segment (closest cluster)
            best_deficit_seg = None
            best_distance = float('inf')

            for deficit_seg in deficit.keys():
                if deficit[deficit_seg] > 0:
                    dist = df_work.loc[idx, f'dist_to_{deficit_seg}']
                    if dist < best_distance:
                        best_distance = dist
                        best_deficit_seg = deficit_seg

            if best_deficit_seg is not None:
                # Perform reallocation
                adjusted_predictions[idx] = best_deficit_seg
                deficit[best_deficit_seg] -= 1

                # Log reallocation
                reallocation_log.append({
                    'index': idx,
                    'product_id': df.iloc[idx]['product_id'] if 'product_id' in df.columns else idx,
                    'from': old_seg,
                    'to': best_deficit_seg,
                    'distance_to_new': best_distance,
                    'distance_to_old': df_work.loc[idx, f'dist_to_{old_seg}'],
                    'distance_ratio': best_distance / (df_work.loc[idx, f'dist_to_{old_seg}'] + 1e-10)
                })

                if deficit[best_deficit_seg] == 0:
                    del deficit[best_deficit_seg]

    # Verify final distribution
    final_counts = pd.Series(adjusted_predictions).value_counts().to_dict()
    print("\nFinal distribution after adjustment:")
    for seg in ['KVI', 'SD', 'PG']:
        count = final_counts.get(seg, 0)
        target = target_counts[seg]
        match = "✓" if count == target else "✗"
        print(f"  {match} {seg}: {count:5d} (target: {target:5d})")
    print()

    # Statistics on reallocated samples
    if reallocation_log:
        realloc_df = pd.DataFrame(reallocation_log)
        print("Reallocation statistics:")
        print(f"  Mean distance to new cluster: {realloc_df['distance_to_new'].mean():.4f}")
        print(f"  Mean distance to old cluster: {realloc_df['distance_to_old'].mean():.4f}")
        print(f"  Mean distance ratio (new/old): {realloc_df['distance_ratio'].mean():.4f}")
        print(f"  Median distance ratio: {realloc_df['distance_ratio'].median():.4f}")

    reallocation_info = {
        'n_reallocated': len(reallocation_log),
        'reallocation_log': reallocation_log
    }

    print(f"\nTotal samples reallocated: {len(reallocation_log)}")

    return adjusted_predictions, reallocation_info


def main():
    """
    Main prediction pipeline.

    Steps:
    1. Load raw data (ALL rows)
    2. Preprocess (adjust violations, no filtering)
    3. Engineer features
    4. Load scaler and transform
    5. Load all 3 models
    6. Generate predictions from each model
    7. Ensemble via voting
    8. Apply distribution reallocation
    9. Save final predictions CSV
    """

    print("\n" + "=" * 80)
    print("PREDICTION PIPELINE")
    print("=" * 80 + "\n")

    # Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "data.parquet")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Target distribution
    TARGET_DISTRIBUTION = {
        'KVI': 0.10,
        'SD': 0.30,
        'PG': 0.60
    }

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data path: {DATA_PATH}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

    # ========================================================================
    # STEP 1: Load raw data (ALL rows)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)

    df_raw = load_data(DATA_PATH)
    print(f"Total samples to predict: {len(df_raw)}")

    # ========================================================================
    # STEP 2: Preprocess data (predict mode - adjust, no filtering)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: PREPROCESS DATA (PREDICT MODE)")
    print("=" * 80)

    df_clean = preprocess_data(df_raw, mode='predict')

    # ========================================================================
    # STEP 3: Feature engineering
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 80)

    df_engineered, engineered_feature_names = engineer_features(df_clean)

    # Get all numeric features
    numeric_features = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
    if 'product_id' in numeric_features:
        numeric_features.remove('product_id')

    print(f"\nTotal numeric features: {len(numeric_features)}")

    # ========================================================================
    # STEP 4: Load models and metadata
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: LOAD TRAINED MODELS")
    print("=" * 80)

    # Load feature metadata
    metadata = load_model(os.path.join(MODELS_DIR, "feature_metadata.pkl"))
    top_4_features = metadata['top_4_features']

    print(f"\nTop 4 features for clustering: {top_4_features}")

    # Load scaler
    scaler = load_model(os.path.join(MODELS_DIR, "robust_scaler.pkl"))

    # Load Decision Tree
    dt_model = load_model(os.path.join(MODELS_DIR, "decision_tree.pkl"))

    # Load Hierarchical Clustering
    hierarchical_bundle = load_model(os.path.join(MODELS_DIR, "monotonic_clustering.pkl"))
    hierarchical_knn = hierarchical_bundle['knn_predictor']
    hierarchical_mapping = hierarchical_bundle['cluster_mapping']

    # Load GMM
    gmm_bundle = load_model(os.path.join(MODELS_DIR, "gmm_model.pkl"))
    gmm_model = gmm_bundle['model']
    gmm_mapping = gmm_bundle['cluster_mapping']

    print("\nAll models loaded successfully")

    # ========================================================================
    # STEP 5: Scale features
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: SCALE FEATURES")
    print("=" * 80)

    X = df_engineered[numeric_features].copy()
    X_scaled = transform_features(scaler, X, feature_names=numeric_features)

    # Extract top 4 features for clustering
    X_scaled_top4 = X_scaled[top_4_features].values
    X_scaled_all = X_scaled.values

    print(f"Scaled data shape (all features): {X_scaled_all.shape}")
    print(f"Scaled data shape (top 4 features): {X_scaled_top4.shape}")

    # ========================================================================
    # STEP 6: Generate predictions from all 3 models
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: GENERATE PREDICTIONS FROM ALL MODELS")
    print("=" * 80 + "\n")

    # Model 1: Decision Tree (uses all features)
    print("1. Decision Tree predictions...")
    dt_predictions = dt_model.predict(X_scaled_all)
    print(f"   Distribution: {pd.Series(dt_predictions).value_counts().sort_index().to_dict()}")

    # Model 2: Hierarchical Clustering (uses top 4 features)
    print("\n2. Hierarchical Clustering predictions...")
    hierarchical_clusters = hierarchical_knn.predict(X_scaled_top4)
    hierarchical_predictions = np.array([hierarchical_mapping[c] for c in hierarchical_clusters])
    print(f"   Distribution: {pd.Series(hierarchical_predictions).value_counts().sort_index().to_dict()}")

    # Model 3: GMM (uses top 4 features)
    print("\n3. GMM predictions...")
    gmm_clusters = gmm_model.predict(X_scaled_top4)
    gmm_predictions = np.array([gmm_mapping[c] for c in gmm_clusters])
    print(f"   Distribution: {pd.Series(gmm_predictions).value_counts().sort_index().to_dict()}")

    # ========================================================================
    # STEP 7: Ensemble predictions via voting
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: ENSEMBLE PREDICTIONS (VOTING)")
    print("=" * 80 + "\n")

    # Create ensemble dataframe
    ensemble_df = pd.DataFrame({
        'product_id': df_engineered['product_id'].values,
        'dt_pred': dt_predictions,
        'hierarchical_pred': hierarchical_predictions,
        'gmm_pred': gmm_predictions
    })

    # Voting: mode of 3 predictions
    ensemble_df['initial_prediction'] = ensemble_df[['dt_pred', 'hierarchical_pred', 'gmm_pred']].mode(axis=1)[0]

    initial_predictions = ensemble_df['initial_prediction'].values

    print("Ensemble (voting) distribution:")
    print(pd.Series(initial_predictions).value_counts().sort_index())
    print()

    # ========================================================================
    # STEP 8: Apply distribution reallocation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: DISTRIBUTION REALLOCATION")
    print("=" * 80)

    print(f"\nTarget distribution:")
    for seg, prop in TARGET_DISTRIBUTION.items():
        print(f"  {seg}: {prop * 100:.0f}%")
    print()

    # Use GMM for reallocation (best model from notebook)
    adjusted_predictions, reallocation_info = adjust_predictions_with_cluster_distances(
        df=df_engineered,
        X_scaled=X_scaled_top4,
        clustering_model=gmm_model,
        cluster_mapping=gmm_mapping,
        initial_predictions=initial_predictions,
        target_distribution=TARGET_DISTRIBUTION,
        top_features=top_4_features,
        model_type='gmm'
    )

    # ========================================================================
    # STEP 9: Save final predictions
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: SAVE PREDICTIONS")
    print("=" * 80 + "\n")

    # Create final results dataframe
    results_df = pd.DataFrame({
        'product_id': df_engineered['product_id'].values,
        'predicted_label': adjusted_predictions
    })

    # Save to CSV
    output_path = os.path.join(RESULTS_DIR, "final_predictions.csv")
    results_df.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print(f"Total predictions: {len(results_df)}")

    # Final distribution
    print("\nFinal prediction distribution:")
    print(results_df['predicted_label'].value_counts().sort_index())
    print("\nProportions:")
    print(results_df['predicted_label'].value_counts(normalize=True).sort_index().round(4))

    # ========================================================================
    # PREDICTION COMPLETE
    # ========================================================================
    print("\n" + "=" * 80)
    print("PREDICTION PIPELINE COMPLETE")
    print("=" * 80 + "\n")

    print(f"Output file: {output_path}")
    print(f"Rows predicted: {len(results_df)}")
    print(f"Samples reallocated: {reallocation_info['n_reallocated']}")


if __name__ == "__main__":
    main()