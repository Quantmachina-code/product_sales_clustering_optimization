import pickle
import os
from pathlib import Path


def save_model(model, filepath):
    """
    Save model to disk using pickle.

    Parameters:
    -----------
    model : object
        Model or transformer to save
    filepath : str
        Full path including filename
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load model from disk.

    Parameters:
    -----------
    filepath : str
        Full path to model file

    Returns:
    --------
    model : object
        Loaded model or transformer
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # Test save/load functionality
    from sklearn.tree import DecisionTreeClassifier

    # Create dummy model
    test_model = DecisionTreeClassifier(max_depth=3, random_state=42)

    # Test paths
    test_dir = "models/test"
    test_path = f"{test_dir}/test_model.pkl"

    # Test save
    save_model(test_model, test_path)

    # Test load
    loaded_model = load_model(test_path)

    # Verify
    assert loaded_model.max_depth == 3
    assert loaded_model.random_state == 42

    # Cleanup
    os.remove(test_path)
    os.rmdir(test_dir)

    print("✓ io_utils.py test passed")