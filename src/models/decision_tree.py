import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


def train_decision_tree(X_labeled, y_labeled, feature_names, max_depth=3,
                        max_features=4, min_samples_split=5, min_samples_leaf=2,
                        random_state=42):
    """
    Train Decision Tree on labeled samples.

    Parameters:
    -----------
    X_labeled : pd.DataFrame or np.ndarray
        Scaled features from labeled samples
    y_labeled : pd.Series or np.ndarray
        True labels (KVI, SD, PG)
    feature_names : list
        List of feature column names
    max_depth : int
        Maximum tree depth for interpretability
    max_features : int
        Number of top features to consider
    min_samples_split : int
        Minimum samples required to split node
    min_samples_leaf : int
        Minimum samples in leaf node
    random_state : int
        Random seed

    Returns:
    --------
    dt_model : DecisionTreeClassifier
        Trained decision tree
    feature_importances_df : pd.DataFrame
        Feature importances sorted
    top_features : list
        Top features with non-zero importance
    """
    print("\n=== DECISION TREE TRAINING ===\n")
    print(f"Training samples: {X_labeled.shape[0]}")
    print(f"Features: {X_labeled.shape[1]}")

    # Train decision tree
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    dt_model.fit(X_labeled, y_labeled)

    # Get feature importances
    feature_importances_df = pd.DataFrame({
        'feature': feature_names,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Feature Importances (top 10):")
    print(feature_importances_df.head(10))

    # Extract top features (non-zero importance)
    top_features = feature_importances_df[feature_importances_df['importance'] > 0]['feature'].tolist()
    print(f"\nTop features used: {len(top_features)}")
    print(f"Top 4 features: {top_features[:4]}")

    # Evaluate on training data
    y_pred = dt_model.predict(X_labeled)

    print("\n=== DECISION TREE PERFORMANCE (labeled data) ===\n")
    print(classification_report(y_labeled, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_labeled, y_pred)
    print(cm)

    return dt_model, feature_importances_df, top_features


if __name__ == "__main__":
    # Test decision tree training
    from sklearn.datasets import make_classification

    # Create synthetic labeled data
    X, y = make_classification(
        n_samples=54,
        n_features=20,
        n_informative=10,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(20)]
    labels = np.array(['KVI', 'SD', 'PG'])
    y_labeled = labels[y]

    # Train decision tree
    dt_model, importances, top_features = train_decision_tree(
        X, y_labeled, feature_names,
        max_depth=3, max_features=4, random_state=42
    )

    # Verify
    assert dt_model.max_depth == 3
    assert len(top_features) > 0
    assert len(importances) == 20

    # Test prediction
    y_pred = dt_model.predict(X[:5])
    assert len(y_pred) == 5

    print("\n✓ decision_tree.py test passed")