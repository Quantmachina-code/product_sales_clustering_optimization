import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score


def create_cluster_mapping_by_pressure(df_train, cluster_labels, top_features=None):
    """
    Map clusters to business segments based on competitive_pressure.

    Logic from notebook:
    - Lowest pressure → PG (Profit Generators)
    - Highest pressure → KVI (Key Value Items)
    - Middle pressure → SD (Sales Drivers)

    Parameters:
    -----------
    df_train : pd.DataFrame
        Training dataframe with engineered features
    cluster_labels : np.ndarray
        Cluster assignments for training data
    top_features : list, optional
        Feature subset used for clustering

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


def train_hierarchical_clustering(X_train, df_train, linkage='ward', metric='euclidean',
                                  top_features=None, random_state=42):
    """
    Train Hierarchical clustering model.

    Best parameters from notebook: linkage='ward', metric='euclidean'

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Scaled training features
    df_train : pd.DataFrame
        Training dataframe with engineered features (for mapping)
    linkage : str
        Linkage criterion
    metric : str
        Distance metric
    top_features : list, optional
        Feature subset to use (if None, use all)
    random_state : int
        Random seed (not used by AgglomerativeClustering but kept for consistency)

    Returns:
    --------
    hierarchical_model : AgglomerativeClustering
        Trained clustering model
    knn_predictor : KNeighborsClassifier
        KNN model for prediction (hierarchical has no predict method)
    cluster_mapping : dict
        Cluster to segment mapping
    train_labels : np.ndarray
        Cluster assignments for training data
    """
    print("\n=== HIERARCHICAL CLUSTERING TRAINING ===\n")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Linkage: {linkage}, Metric: {metric}")

    # Select features if specified
    if top_features is not None:
        if isinstance(X_train, pd.DataFrame):
            X_train_subset = X_train[top_features].values
        else:
            # Assume top_features are indices
            X_train_subset = X_train[:, top_features]
        print(f"Using top {len(top_features)} features")
    else:
        X_train_subset = X_train if not isinstance(X_train, pd.DataFrame) else X_train.values

    # Train hierarchical clustering
    hierarchical_model = AgglomerativeClustering(
        n_clusters=3,
        linkage=linkage,
        metric=metric
    )

    train_labels = hierarchical_model.fit_predict(X_train_subset)

    # Calculate metrics
    train_silhouette = silhouette_score(X_train_subset, train_labels)
    train_db = davies_bouldin_score(X_train_subset, train_labels)

    print(f"\nTraining metrics:")
    print(f"  Silhouette Score: {train_silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {train_db:.4f}")

    # Create cluster to segment mapping
    cluster_mapping = create_cluster_mapping_by_pressure(df_train, train_labels, top_features)

    # Train KNN predictor (hierarchical has no predict method)
    knn_predictor = KNeighborsClassifier(n_neighbors=5)
    knn_predictor.fit(X_train_subset, train_labels)

    print("\nKNN predictor trained for inference on new data")

    # Distribution
    print("\nCluster distribution:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        segment = cluster_mapping[cluster_id]
        print(f"  Cluster {cluster_id} ({segment}): {count} ({count / len(train_labels) * 100:.2f}%)")

    return hierarchical_model, knn_predictor, cluster_mapping, train_labels


if __name__ == "__main__":
    # Test hierarchical clustering
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

    # Train clustering
    hierarchical_model, knn_predictor, cluster_mapping, train_labels = train_hierarchical_clustering(
        X, df_mock,
        linkage='ward',
        metric='euclidean',
        random_state=42
    )

    # Verify
    assert len(train_labels) == 200
    assert len(np.unique(train_labels)) == 3
    assert len(cluster_mapping) == 3
    assert set(cluster_mapping.values()) == {'KVI', 'SD', 'PG'}

    # Test prediction with KNN
    y_pred = knn_predictor.predict(X[:5])
    assert len(y_pred) == 5

    print("\n✓ clustering.py test passed")