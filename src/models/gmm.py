import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score


def create_cluster_mapping_by_pressure_gmm(df_train, cluster_labels):
    """
    Map GMM clusters to business segments based on competitive_pressure.

    Parameters:
    -----------
    df_train : pd.DataFrame
        Training dataframe with engineered features
    cluster_labels : np.ndarray
        Cluster assignments for training data

    Returns:
    --------
    cluster_mapping : dict
        Mapping from cluster ID to segment label
    """
    # Add cluster labels to dataframe
    df_with_clusters = df_train.copy()
    df_with_clusters['cluster'] = cluster_labels

    # Calculate mean competitive_pressure per cluster
    cluster_pressure = df_with_clusters.groupby('cluster')['competitive_pressure'].mean().sort_values()

    # Mapping: lowest pressure = PG, highest = KVI, middle = SD
    sorted_clusters = cluster_pressure.index.tolist()

    cluster_mapping = {
        sorted_clusters[0]: 'PG',  # Lowest pressure
        sorted_clusters[1]: 'SD',  # Middle pressure
        sorted_clusters[2]: 'KVI'  # Highest pressure
    }

    print("\nCluster to Segment Mapping (by competitive_pressure):")
    for cluster_id, segment in cluster_mapping.items():
        pressure = cluster_pressure[cluster_id]
        count = (cluster_labels == cluster_id).sum()
        print(f"  Cluster {cluster_id} → {segment} (pressure={pressure:.4f}, n={count})")

    return cluster_mapping


def train_gmm(X_train, df_train, covariance_type='full', n_init=10,
              max_iter=200, top_features=None, random_state=42):
    """
    Train Gaussian Mixture Model with optimized parameters.

    Best parameters from notebook hyperoptimization:
    - covariance_type: 'full'
    - n_init: 10
    - max_iter: 200

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Scaled training features
    df_train : pd.DataFrame
        Training dataframe with engineered features (for mapping)
    covariance_type : str
        Type of covariance parameters
    n_init : int
        Number of initializations
    max_iter : int
        Maximum iterations
    top_features : list, optional
        Feature subset to use (if None, use all)
    random_state : int
        Random seed

    Returns:
    --------
    gmm_model : GaussianMixture
        Trained GMM model
    cluster_mapping : dict
        Cluster to segment mapping
    train_labels : np.ndarray
        Cluster assignments for training data
    """
    print("\n=== GAUSSIAN MIXTURE MODEL TRAINING ===\n")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Covariance type: {covariance_type}")
    print(f"n_init: {n_init}, max_iter: {max_iter}")

    # Select features if specified
    if top_features is not None:
        if isinstance(X_train, pd.DataFrame):
            X_train_subset = X_train[top_features].values
        else:
            X_train_subset = X_train[:, top_features]
        print(f"Using top {len(top_features)} features")
    else:
        X_train_subset = X_train if not isinstance(X_train, pd.DataFrame) else X_train.values

    # Train GMM
    gmm_model = GaussianMixture(
        n_components=3,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )

    gmm_model.fit(X_train_subset)
    train_labels = gmm_model.predict(X_train_subset)

    # Calculate metrics
    train_silhouette = silhouette_score(X_train_subset, train_labels)
    train_db = davies_bouldin_score(X_train_subset, train_labels)

    print(f"\nTraining metrics:")
    print(f"  Silhouette Score: {train_silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {train_db:.4f}")
    print(f"  Converged: {gmm_model.converged_}")
    print(f"  Log-likelihood: {gmm_model.lower_bound_:.2f}")

    # Create cluster to segment mapping
    cluster_mapping = create_cluster_mapping_by_pressure_gmm(df_train, train_labels)

    # Distribution
    print("\nCluster distribution:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        segment = cluster_mapping[cluster_id]
        print(f"  Cluster {cluster_id} ({segment}): {count} ({count / len(train_labels) * 100:.2f}%)")

    return gmm_model, cluster_mapping, train_labels


if __name__ == "__main__":
    # Test GMM training
    from sklearn.datasets import make_classification

    # Create synthetic data
    X, _ = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_clusters_per_class=1,
        random_state=42
    )

    # Create mock dataframe with competitive_pressure
    df_mock = pd.DataFrame({
        'product_id': range(200),
        'competitive_pressure': np.random.uniform(0.1, 2.0, 200)
    })

    # Train GMM
    gmm_model, cluster_mapping, train_labels = train_gmm(
        X, df_mock,
        covariance_type='full',
        n_init=10,
        max_iter=200,
        random_state=42
    )

    # Verify
    assert len(train_labels) == 200
    assert len(np.unique(train_labels)) == 3
    assert len(cluster_mapping) == 3
    assert set(cluster_mapping.values()) == {'KVI', 'SD', 'PG'}
    assert gmm_model.converged_

    # Test prediction
    y_pred = gmm_model.predict(X[:5])
    assert len(y_pred) == 5

    print("\n✓ gmm.py test passed")