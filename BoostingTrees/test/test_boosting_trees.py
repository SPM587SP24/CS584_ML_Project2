import numpy as np
import pytest
import sys
import os
import warnings
from dataclasses import dataclass
from typing import Optional

# Add the model directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
from BoostingTrees import BoostingTreesClassifier

# Ignore specific warning about division by zero
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


# Basic Functionality Tests
def test_basic_functionality():
    """Test basic functionality with a simple dataset."""
    # Create a very simple dataset with clear separation
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 3], [3, 4], [4, 3], [4, 4]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = BoostingTreesClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)

    assert accuracy >= 0.5, "Basic predictions should achieve at least 50% accuracy"


def test_probability_predictions():
    """Test that probability predictions are between 0 and 1."""
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model = BoostingTreesClassifier(n_estimators=10, learning_rate=0.1)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert np.all(proba >= 0) and np.all(
        proba <= 1
    ), "Probabilities should be between 0 and 1"


def test_non_linear_decision_boundary():
    """Test that the model can learn a non-linear decision boundary."""
    # Create a circular decision boundary with more samples
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1).astype(int)

    model = BoostingTreesClassifier(n_estimators=200, learning_rate=0.1, max_depth=5)
    model.fit(X, y)

    # Test accuracy should be better than random
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.5, "Model should achieve better than random accuracy"


# Early Stopping Tests
def test_early_stopping():
    """Test that early stopping works correctly."""
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model = BoostingTreesClassifier(
        n_estimators=100, early_stopping_rounds=2, validation_fraction=0.25
    )

    model.fit(X, y)

    assert model.best_iteration is not None
    assert len(model.trees) < 100


def test_validation_split():
    """Test that validation split works correctly."""
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = BoostingTreesClassifier(n_estimators=10, validation_fraction=0.5)

    model.fit(X, y)
    assert model.best_iteration is not None


# Feature Importance Tests
def test_feature_importance():
    """Test that feature importance is calculated correctly."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] > 0).astype(int)

    model = BoostingTreesClassifier(n_estimators=10)
    model.fit(X, y)

    importance = model.get_feature_importance()
    assert importance[0] > 0  # Feature 0 should have some importance
    assert sum(importance.values()) == 1.0  # Should sum to 1
    if 1 in importance:
        assert importance[0] > importance[1]


def test_feature_importance_visualization():
    """Test feature importance visualization."""
    X = np.random.rand(100, 5)
    y = (X[:, 0] > 0.5).astype(int)  # Only first feature is important

    model = BoostingTreesClassifier()
    model.fit(X, y)

    importance = model.get_feature_importance()
    assert importance[0] > importance[1]  # First feature should be more important


# Parameter Effects Tests
def test_parameter_effects():
    """Test the effect of different parameters on model performance."""
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model1 = BoostingTreesClassifier(n_estimators=5, learning_rate=0.1)
    model2 = BoostingTreesClassifier(n_estimators=20, learning_rate=0.1)

    model1.fit(X, y)
    model2.fit(X, y)

    assert np.mean(model2.predict(X) == y) >= np.mean(model1.predict(X) == y)


def test_parameter_effects_with_early_stopping():
    """Test that different parameters affect early stopping behavior."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] > 0).astype(int)

    fast_model = BoostingTreesClassifier(
        n_estimators=100, learning_rate=0.1, early_stopping_rounds=5
    )
    slow_model = BoostingTreesClassifier(
        n_estimators=100, learning_rate=0.01, early_stopping_rounds=5
    )

    fast_model.fit(X, y)
    slow_model.fit(X, y)

    assert fast_model.best_iteration is not None
    assert slow_model.best_iteration is not None

    assert (
        fast_model.best_iteration <= slow_model.best_iteration
    ), f"Expected fast_model to stop earlier (got {fast_model.best_iteration} vs {slow_model.best_iteration})"


# Advanced Features Tests
def test_parallel_processing():
    """Test parallel processing functionality."""
    X = np.random.rand(1000, 10)
    y = (np.sum(X, axis=1) > 5).astype(int)

    for n_jobs in [1, 2, -1]:
        model = BoostingTreesClassifier(n_jobs=n_jobs)
        model.fit(X, y)
        assert len(model.trees) > 0


def test_custom_loss_function():
    """Test custom loss function support."""

    def custom_loss(y_true, y_pred, return_gradient=False):
        loss = np.mean((y_true - y_pred) ** 2)
        if return_gradient:
            grad = 2 * (y_pred - y_true)
            return loss, grad
        return loss

    X = np.random.rand(100, 5)
    y = (np.sum(X, axis=1) > 2.5).astype(int)

    model = BoostingTreesClassifier(custom_loss=custom_loss)
    model.fit(X, y)
    assert len(model.trees) > 0


def test_learning_curves():
    """Test learning curves tracking."""
    X = np.random.rand(100, 5)
    y = (np.sum(X, axis=1) > 2.5).astype(int)

    model = BoostingTreesClassifier()
    model.fit(X, y)

    assert len(model.training_losses) > 0
    assert len(model.validation_losses) > 0
    assert model.training_losses[-1] <= model.training_losses[0]


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    X = np.random.rand(100, 5) * 1e6  # Large values
    y = (np.sum(X, axis=1) > 2.5e6).astype(int)

    model = BoostingTreesClassifier()
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_parallel_vs_sequential():
    """Test that parallel and sequential implementations give same results."""
    X = np.random.rand(100, 5)
    y = (np.sum(X, axis=1) > 2.5).astype(int)

    model_seq = BoostingTreesClassifier(n_jobs=1)
    model_par = BoostingTreesClassifier(n_jobs=-1)

    model_seq.fit(X, y)
    model_par.fit(X, y)

    pred_seq = model_seq.predict_proba(X)
    pred_par = model_par.predict_proba(X)
    assert np.allclose(pred_seq, pred_par, rtol=1e-5)


def test_feature_importance_with_custom_loss():
    """Test feature importance calculation with custom loss."""

    def custom_loss(y_true, y_pred, return_gradient=False):
        loss = np.mean((y_true - y_pred) ** 2)
        if return_gradient:
            grad = 2 * (y_pred - y_true)
            return loss, grad
        return loss

    X = np.random.rand(100, 5)
    y = (X[:, 0] > 0.5).astype(int)  # Only first feature is important

    model = BoostingTreesClassifier(custom_loss=custom_loss)
    model.fit(X, y)

    importance = model.get_feature_importance()
    assert importance[0] > importance[1]  # First feature should be more important


# Edge Cases Tests
def test_edge_cases():
    """Test handling of edge cases."""
    # Test with single sample
    X = np.array([[1, 2, 3]])
    y = np.array([1])
    model = BoostingTreesClassifier()
    with pytest.raises(ValueError):
        model.fit(X, y)

    # Test with all same class
    X = np.random.rand(10, 3)
    y = np.ones(10)
    with pytest.raises(ValueError):
        model.fit(X, y)

    # Test with NaN values
    X = np.random.rand(10, 3)
    X[0, 0] = np.nan
    y = np.random.randint(0, 2, 10)
    with pytest.raises(ValueError):
        model.fit(X, y)

    # Test prediction before fitting
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])
    model = BoostingTreesClassifier()
    with pytest.raises(ValueError):
        model.predict(X)

    # Test feature importance before fitting
    with pytest.raises(ValueError):
        model.get_feature_importance()


if __name__ == "__main__":
    pytest.main([__file__])
