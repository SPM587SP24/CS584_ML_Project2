import numpy as np
from typing import List, Optional, Union, Dict, Callable, Tuple
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import os


@dataclass
class DecisionTree:
    """Decision tree node class."""

    is_leaf: bool = False
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["DecisionTree"] = None
    right: Optional["DecisionTree"] = None
    value: Optional[float] = None
    gain: float = 0.0


class BoostingTreesClassifier:
    """Boosting Classifier with additional features."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        early_stopping_rounds: int = 5,
        validation_fraction: float = 0.1,
        n_jobs: int = -1,
        custom_loss: Optional[Callable] = None,
        subsample: float = 1.0
    ):
        """
        Initialize the enhanced   boosting classifier.

        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages to perform
        learning_rate : float
            Learning rate shrinks the contribution of each tree
        max_depth : int
            Maximum depth of the individual regression estimators
        min_samples_split : int
            Minimum number of samples required to split an internal node
        early_stopping_rounds : int
            Number of rounds without improvement to stop training
        validation_fraction : float
            Fraction of training data to use for validation
        n_jobs : int
            Number of parallel jobs to run (-1 means using all processors)
        custom_loss : Optional[Callable]
            Custom loss function to use instead of log loss
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.n_jobs = n_jobs
        self.custom_loss = custom_loss
        self.subsample = subsample
        self.trees: List[DecisionTree] = []
        self.initial_prediction: Optional[float] = None
        self.feature_importance: Dict[int, float] = defaultdict(float)
        self.best_iteration: Optional[int] = None
        self.validation_losses: List[float] = []
        self.training_losses: List[float] = []
        self.n_features_: Optional[int] = None  # Initialize n_features_ as None
        self.n_classes_ = None  # Number of unique classes (set in fit)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid function with numerical stability."""
        return np.clip(1 / (1 + np.exp(-x)), 1e-15, 1 - 1e-15)

    def _log_loss(self, y: np.ndarray, p: np.ndarray) -> float:
        """Compute log loss with numerical stability."""
        if p.ndim == 2:        # if p is shape (n_samples, 2)
            p = p[:, 1]         # use class 1 probability only
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def _log_loss_gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Compute   of log loss."""
        return y - p

    def _calculate_loss(self, y: np.ndarray, p: np.ndarray) -> float:
        """Calculate loss using either custom or default loss function."""
        if self.custom_loss is not None:
            return self.custom_loss(y, p)
        return self._log_loss(y, p)

    def _calculate_gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Calculate   using either custom or default loss function."""
        if self.custom_loss is not None:
            # Assume custom loss function returns both loss and  
            _, grad = self.custom_loss(y, p, return_gradient=True)
            return grad
        return self._log_loss_gradient(y, p)

    def _calculate_gain(
        self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
    ) -> float:
        """Calculate the gain for a split.

        Parameters:
        -----------
        y : np.ndarray
            The target values for all samples
        y_left : np.ndarray
            The target values for samples in the left split
        y_right : np.ndarray
            The target values for samples in the right split

        Returns:
        --------
        float
            The gain for this split
        """

        def calculate_impurity(y_subset):
            if len(y_subset) == 0:
                return 0
            p = np.mean(y_subset)
            if p == 0 or p == 1:
                return 0
            return -p * np.log(p) - (1 - p) * np.log(1 - p)

        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        # Calculate variance reduction (more suitable for regression-like residuals)
        parent_var = np.var(y) if len(y) > 0 else 0
        left_var = np.var(y_left) if len(y_left) > 0 else 0
        right_var = np.var(y_right) if len(y_right) > 0 else 0

        # Weight the variance reduction by the size of the splits
        variance_reduction = (
            parent_var - (n_left / n) * left_var - (n_right / n) * right_var
        )

        # Calculate entropy-based gain
        impurity_parent = calculate_impurity(y)
        impurity_left = calculate_impurity(y_left)
        impurity_right = calculate_impurity(y_right)
        entropy_gain = (
            impurity_parent
            - (n_left / n) * impurity_left
            - (n_right / n) * impurity_right
        )

        # Combine both metrics with more weight on variance reduction
        gain = 0.7 * variance_reduction + 0.3 * entropy_gain
        return max(gain, 0)  # Ensure non-negative gain

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity (MSE) for a set of target values."""
        if len(y) == 0:
            return 0.0
        return np.mean((y - np.mean(y)) ** 2)

    def _find_best_split_for_features(
        self, X: np.ndarray, y: np.ndarray, features: List[int]
    ) -> Optional[Tuple[float, dict]]:
        """Find the best split for a subset of features."""
        best_gain = float("-inf")
        best_split_info = None

        for feature_idx in features:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            if len(unique_values) <= 1:
                continue

            # Consider potential split points
            split_candidates = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in split_candidates:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if (
                    np.sum(left_mask) < self.min_samples_split
                    or np.sum(right_mask) < self.min_samples_split
                ):
                    continue

                # Calculate gain
                left_y = y[left_mask]
                right_y = y[right_mask]

                left_pred = np.mean(left_y)
                right_pred = np.mean(right_y)

                current_mse = np.mean((y - np.mean(y)) ** 2)
                left_mse = np.mean((left_y - left_pred) ** 2)
                right_mse = np.mean((right_y - right_pred) ** 2)

                # Weighted MSE reduction
                n_left = len(left_y)
                n_right = len(right_y)
                n_total = len(y)

                gain = current_mse - (
                    (n_left / n_total) * left_mse + (n_right / n_total) * right_mse
                )

                if gain > best_gain:
                    best_gain = gain
                    best_split_info = {
                        "feature_idx": feature_idx,
                        "threshold": threshold,
                        "left_indices": np.where(left_mask)[0],
                        "right_indices": np.where(right_mask)[0],
                    }

        if best_split_info is None:
            return None

        return best_gain, best_split_info

    def _fit_tree_parallel(
        self, X: np.ndarray, y: np.ndarray, n_jobs: int = -1
    ) -> DecisionTree:
        """Fit a decision tree using parallel processing for finding the best split."""
        if len(y) <= self.min_samples_split:
            leaf = DecisionTree(is_leaf=True, value=np.mean(y))
            return leaf

        # Handle n_jobs parameter
        n_jobs = n_jobs if n_jobs > 0 else os.cpu_count() or 1
        n_jobs = min(n_jobs, os.cpu_count() or 1)
        n_jobs = min(n_jobs, X.shape[1])  # Can't use more workers than features

        # Create feature chunks for parallel processing
        n_features = X.shape[1]
        chunk_size = max(1, n_features // n_jobs)
        feature_chunks = [
            list(range(i, min(i + chunk_size, n_features)))
            for i in range(0, n_features, chunk_size)
        ]

        # Initialize variables for best split
        best_gain = float("-inf")
        best_feature_idx = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None

        # Find best split across all features in parallel
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            split_results = list(
                executor.map(
                    lambda features: self._find_best_split_for_features(X, y, features),
                    feature_chunks,
                )
            )

        # Process results from parallel execution
        for result in split_results:
            if result is not None:
                gain, split_info = result
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = split_info["feature_idx"]
                    best_threshold = split_info["threshold"]
                    best_left_indices = split_info["left_indices"]
                    best_right_indices = split_info["right_indices"]

        # If no valid split found, create a leaf node
        if best_feature_idx is None:
            return DecisionTree(is_leaf=True, value=np.mean(y))

        # Create the split node
        node = DecisionTree(
            feature_idx=best_feature_idx, threshold=best_threshold, gain=best_gain
        )

        # Recursively build left and right subtrees
        left_X = X[best_left_indices]
        left_y = y[best_left_indices]
        right_X = X[best_right_indices]
        right_y = y[best_right_indices]

        node.left = self._fit_tree_parallel(left_X, left_y, n_jobs=1)
        node.right = self._fit_tree_parallel(right_X, right_y, n_jobs=1)

        return node

    def _fit_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTree:
        """Recursively build a decision tree."""
        # Check stopping conditions
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return DecisionTree(
                feature_idx=0,
                threshold=0.0,
                left_value=np.mean(y),
                right_value=np.mean(y),
                gain=0.0,
            )

        # Find best split
        best_split = self._find_best_split_for_features(X, y, np.arange(X.shape[1]))

        if best_split is None:
            return DecisionTree(
                feature_idx=0,
                threshold=0.0,
                left_value=np.mean(y),
                right_value=np.mean(y),
                gain=0.0,
            )

        # Unpack all 6 values from the best split
        gain, split_info = best_split

        # Update feature importance with the weighted gain
        self.feature_importance[split_info["feature_idx"]] += (
            gain * len(y) / (self.max_depth + 1)
        )

        # Create the tree
        tree = DecisionTree(
            feature_idx=split_info["feature_idx"],
            threshold=split_info["threshold"],
            gain=gain,
        )

        # Recursively build subtrees if needed
        if depth + 1 < self.max_depth:
            left_X = X[split_info["left_indices"]]
            left_y = y[split_info["left_indices"]]
            right_X = X[split_info["right_indices"]]
            right_y = y[split_info["right_indices"]]

            tree.left = self._fit_tree(left_X, left_y, depth + 1)
            tree.right = self._fit_tree(right_X, right_y, depth + 1)

        return tree

    def plot_learning_curves(self) -> None:
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, label="Training Loss")
        if self.validation_losses:
            plt.plot(self.validation_losses, label="Validation Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self) -> None:
        """Plot feature importance scores."""
        importance = self.get_feature_importance()
        features = list(importance.keys())
        scores = list(importance.values())

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(features)), scores)
        plt.xticks(range(len(features)), [f"Feature {f}" for f in features])
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.title("Feature Importance")
        plt.grid(True)
        plt.show()

    def get_feature_importance(self) -> Dict[int, float]:
        """Get the feature importance scores.

        Returns:
        --------
        dict
            Dictionary mapping feature indices to their importance scores
        """
        # If no trees have been fitted yet, return zero importance for all features
        if not self.trees:
            if self.n_features_ is None:
                raise ValueError("Model not fitted yet. Call fit() first.")
            return {i: 0.0 for i in range(self.n_features_)}

        # Calculate permutation-based importance
        self._calculate_feature_importance(self.X_train_, self.y_train_)

        # Initialize combined importance for all features
        combined_importance = {i: 0.0 for i in range(self.n_features_)}

        # Combine gain-based and permutation-based importance
        for k in range(self.n_features_):
            # Get gain-based importance (default to 0 if not present)
            gain_importance = self.feature_importance.get(k, 0.0)
            # Get permutation-based importance (default to 0 if not present)
            perm_importance = self.permutation_importance.get(k, 0.0)
            # Combine with more weight on gain-based importance
            combined_importance[k] = 0.8 * gain_importance + 0.2 * perm_importance

        # Normalize the combined importance
        total_importance = sum(combined_importance.values()) + 1e-10
        normalized_importance = {
            k: v / total_importance for k, v in combined_importance.items()
        }

        # Ensure the sum is exactly 1.0
        sum_importance = sum(normalized_importance.values())
        if abs(sum_importance - 1.0) > 1e-10:
            # Adjust the largest value to make the sum exactly 1.0
            max_feature = max(normalized_importance.items(), key=lambda x: x[1])[0]
            normalized_importance[max_feature] += 1.0 - sum_importance

        return normalized_importance

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters:
        -----------
        X : np.ndarray
            Input data, shape (n_samples, n_features)

        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, 2) containing probabilities for each class
        """
        """Predict class probabilities for multi-class."""
        if self.initial_prediction is None or not self.trees:
            raise ValueError("Model not fitted yet. Call fit() first.")

        n_samples = X.shape[0]
        raw_scores = np.full((n_samples, self.n_classes_), self.initial_prediction)

        for k in range(self.n_classes_):
            for tree in self.trees[k]:
                raw_scores[:, k] += self.learning_rate * self._predict_tree(X, tree)

        return self._softmax(raw_scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _predict_tree(self, X: np.ndarray, tree: DecisionTree) -> np.ndarray:
        """Predict using a single decision tree.

        Parameters:
        -----------
        X : np.ndarray
            Input data, shape (n_samples, n_features)
        tree : DecisionTree
            The decision tree to use for prediction

        Returns:
        --------
        np.ndarray
            Predictions for each sample
        """
        predictions = np.zeros(len(X))

        for i, x in enumerate(X):
            current_node = tree
            while True:
                if current_node.is_leaf:
                    predictions[i] = current_node.value
                    break

                if current_node.threshold is None:
                    # If threshold is None, go to left child
                    current_node = current_node.left
                    continue

                if x[current_node.feature_idx] <= current_node.threshold:
                    if current_node.left is None:
                        predictions[i] = current_node.value
                        break
                    current_node = current_node.left
                else:
                    if current_node.right is None:
                        predictions[i] = current_node.value
                        break
                    current_node = current_node.right

        return predictions

    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate feature importance for all features."""
        n_samples = len(X)

        # Initialize permutation importance
        self.permutation_importance = defaultdict(float)
        for i in range(X.shape[1]):
            self.permutation_importance[i] = 0.0

        # For each feature, calculate importance by measuring prediction change
        for feature_idx in range(X.shape[1]):
            # Shuffle the feature values
            X_shuffled = X.copy()
            np.random.shuffle(X_shuffled[:, feature_idx])

            # Calculate predictions with original and shuffled data
            original_pred = self.predict_proba(X)
            shuffled_pred = self.predict_proba(X_shuffled)

            # Sum squared diffs across all classes
            importance = np.mean(np.sum((original_pred - shuffled_pred) ** 2, axis=1))

            # Calculate importance as mean squared difference in predictions
            importance = np.mean((original_pred - shuffled_pred) ** 2)
            self.permutation_importance[feature_idx] = importance

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnhancedGradientBoostingClassifier":
        """
        Fit the enhanced   boosting model.

        Parameters:
        -----------
        X : np.ndarray
            Training data, shape (n_samples, n_features)
        y : np.ndarray
            Target values, shape (n_samples,)

        Returns:
        --------
        self : EnhancedGradientBoostingClassifier
            Fitted estimator
        """
        # Check for single class
        if len(np.unique(y)) == 1:
            raise ValueError("Cannot fit model with single class")

        # Check for NaN values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input contains NaN values")
        # Set number of classes for multi-class classification
        self.n_classes_ = len(np.unique(y))

        # Store training data for feature importance calculation
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_ = X.shape[1]

        # Early stopping variables
        best_val_loss = float("inf")
        best_trees = None
        rounds_without_improvement = 0
        self.best_iteration = 0
        min_delta = 1e-4  # Minimum improvement threshold
        min_val_loss = 1e-10  # Minimum validation loss to consider
        self.training_losses = []
        self.validation_losses = []

        # Split data for validation if needed
        if self.early_stopping_rounds is not None:
            val_size = max(1, int(len(X) * self.validation_fraction))
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]
        else:
            X_train, y_train = X, y

        # One-hot encode labels for multi-class   computation
        Y_train_onehot = np.eye(self.n_classes_)[y_train]
        if self.early_stopping_rounds is not None:
            Y_val_onehot = np.eye(self.n_classes_)[y_val]

        # Initialize predictions with clipped mean for numerical stability
        y_mean = np.clip(np.mean(y_train), 1e-15, 1 - 1e-15)
        self.initial_prediction = np.log(y_mean / (1 - y_mean))
        train_pred = np.full((len(X_train), self.n_classes_), self.initial_prediction)

        if self.early_stopping_rounds is not None:
            val_pred = np.full((len(X_val), self.n_classes_), self.initial_prediction)

        # Reset trees and feature importance
        self.trees = [[] for _ in range(self.n_classes_)]  # one list per class

        self.feature_importance = defaultdict(float)

        # Train trees
        for i in range(self.n_estimators):
            # Compute predicted probabilities via softmax
            train_proba = self._softmax(train_pred)
            residuals = Y_train_onehot - train_proba  # Cross-entropy gradient

            for k in range(self.n_classes_):
                # --- Subsampling ---
                if self.subsample < 1.0:
                    sample_size = int(self.subsample * len(X_train))
                    indices = np.random.choice(len(X_train), sample_size, replace=False)
                    X_sub = X_train[indices]
                    y_sub = residuals[indices, k]
                else:
                    X_sub = X_train
                    y_sub = residuals[:, k]

                # --- Train tree on subsample ---
                tree = self._fit_tree_parallel(X_sub, y_sub)
                self.trees[k].append(tree)

                # --- Update training predictions ---
                train_pred[:, k] += self.learning_rate * self._predict_tree(X_train, tree)

                # --- Update validation predictions if enabled ---
                if self.early_stopping_rounds is not None:
                    val_pred[:, k] += self.learning_rate * self._predict_tree(X_val, tree)

                # Compute training loss
                train_proba = self._softmax(train_pred)
                train_loss = -np.mean(np.sum(Y_train_onehot * np.log(train_proba + 1e-15), axis=1))
                self.training_losses.append(train_loss)

                # Early stopping logic
                if self.early_stopping_rounds is not None:
                    val_proba = self._softmax(val_pred)
                    val_loss = -np.mean(np.sum(Y_val_onehot * np.log(val_proba + 1e-15), axis=1))
                    val_loss = max(val_loss, min_val_loss)
                    self.validation_losses.append(val_loss)

                    if not np.isfinite(val_loss):
                        warnings.warn(f"Invalid validation loss at iteration {i}: {val_loss}")
                        break

                    absolute_improvement = best_val_loss - val_loss
                    denom = best_val_loss + min_val_loss
                    relative_improvement = (
                        absolute_improvement / denom if np.isfinite(denom) and denom > 1e-12 else 0.0
                    )

                    if absolute_improvement > 0 and relative_improvement > min_delta:
                        best_val_loss = val_loss
                        best_trees = [t.copy() for t in self.trees]
                        rounds_without_improvement = 0
                        self.best_iteration = i
                    else:
                        rounds_without_improvement += 1

                    effective_early_stopping = max(
                        self.early_stopping_rounds,
                        int(self.early_stopping_rounds / max(np.sqrt(self.learning_rate), 0.1)),
                    )

                    if rounds_without_improvement >= effective_early_stopping:
                        if best_trees is not None:
                            self.trees = best_trees
                        break


        # Calculate feature importance
        self._calculate_feature_importance(X, y)

        return self
