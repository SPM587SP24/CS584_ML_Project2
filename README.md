# Project 2

## Boosting Trees Implementation

This project implements a Gradient Boosting Trees classifier from first principles, following the methodology described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). The implementation provides a robust and flexible framework for classification tasks with various tuning parameters and advanced features.

## Overview

The BoostingTreesClassifier is a powerful ensemble learning method that combines multiple decision trees to create a strong classifier. It works by:
1. Building trees sequentially, where each new tree tries to correct the errors made by previous trees
2. Using gradient descent to minimize the loss function
3. Combining predictions from all trees with a learning rate to prevent overfitting

## What does the model you have implemented do and when should it be used?

The implemented model is a Gradient Boosting Trees classifier that:

1. **Core Functionality**:
   - Implements gradient boosting for classification tasks
   - Uses decision trees as base learners
   - Supports multi-class classification
   - Implements early stopping to prevent overfitting
   - Calculates feature importance
   - Provides probability estimates for predictions

2. **Key Features**:
   - Multi-class classification support
   - Early stopping to prevent overfitting
   - Feature importance calculation
   - Parallel processing capabilities
   - Custom loss function support
   - Learning curve visualization
   - Numerical stability handling

3. **When to Use**:
   - When dealing with complex classification problems
   - When you need interpretable feature importance
   - When you want to avoid overfitting through early stopping
   - When you need to handle multi-class classification problems
   - When you want to leverage parallel processing for faster training
   - When you have complex, non-linear relationships in your data
   - When you need a model that can handle both numerical and categorical features
   - When you need a model that can be tuned for different trade-offs between bias and variance

## How did you test your model to determine if it is working reasonably correctly?

The model was thoroughly tested through a comprehensive test suite that verifies:

1. **Basic Functionality Tests**:
   - Simple classification tasks with clear separation
   - Probability predictions (ensuring values between 0 and 1)
   - Non-linear decision boundaries (circular decision boundary test)
   - Basic accuracy checks

2. **Advanced Feature Tests**:
   - Early stopping mechanism verification
   - Validation split functionality
   - Feature importance calculation and visualization
   - Parallel processing vs sequential processing
   - Custom loss function implementation
   - Learning curves tracking
   - Numerical stability with extreme values

3. **Edge Case Tests**:
   - Single sample handling
   - Single class detection
   - NaN value detection
   - Pre-fitting prediction attempts
   - Feature importance before fitting
   - Memory usage with large datasets

4. **Performance Tests**:
   - Parameter effects on model performance
   - Early stopping behavior with different parameters
   - Parallel processing efficiency
   - Numerical stability with large values

## What parameters have you exposed to users of your implementation in order to tune performance?

The model exposes several parameters for tuning performance:

1. **Core Parameters**:
   - `n_estimators`: Number of boosting stages (default: 100)
   - `learning_rate`: Shrinks the contribution of each tree (default: 0.1)
   - `max_depth`: Maximum depth of individual trees (default: 3)
   - `min_samples_split`: Minimum samples required to split a node (default: 2)

2. **Advanced Parameters**:
   - `early_stopping_rounds`: Number of rounds without improvement to stop (default: 5)
   - `validation_fraction`: Fraction of training data for validation (default: 0.1)
   - `n_jobs`: Number of parallel jobs (-1 for all processors)
   - `custom_loss`: Custom loss function
   - `subsample`: Fraction of samples to use for each tree (default: 1.0)

### Usage Examples

```python
# Basic usage
from BoostingTrees import BoostingTreesClassifier

# Create and train the model
model = BoostingTreesClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Get feature importance
importance = model.get_feature_importance()

# Visualize learning curves
model.plot_learning_curves()

# Visualize feature importance
model.plot_feature_importance()
```

### Advanced Usage Examples

1. Using Early Stopping:
```python
model = BoostingTreesClassifier(
    n_estimators=100,
    early_stopping_rounds=5,
    validation_fraction=0.2
)
```

2. Using Parallel Processing:
```python
model = BoostingTreesClassifier(n_jobs=-1)  # Use all available processors
```

3. Using Custom Loss Function:
```python
def custom_loss(y_true, y_pred, return_gradient=False):
    loss = np.mean((y_true - y_pred) ** 2)
    if return_gradient:
        grad = 2 * (y_pred - y_true)
        return loss, grad
    return loss

model = BoostingTreesClassifier(custom_loss=custom_loss)
```

## Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

1. **Current Limitations**:
   - Single sample training is not supported (fundamental limitation)
   - Single class detection is not supported (can be worked around)
   - NaN values in input data are not supported (can be worked around)
   - Memory usage can be high with large datasets (can be optimized)
   - Training time can be long with many trees (can be optimized)

2. **Potential Improvements**:
   - Implement sparse matrix support for memory efficiency
   - Add support for categorical features
   - Implement more sophisticated early stopping criteria
   - Add support for sample weights
   - Implement pruning strategies for trees
   - Add support for missing values
   - Implement more advanced feature importance methods

## Project Structure

```
BoostingTrees/
├── model/
│   └── BoostingTrees.py        # Main model implementation
├── test/
│   └── test_boosting_trees.py  # Comprehensive test suite
├── requirements.txt            # Project dependencies
├── README_GIVEN.md            # Project documentation
├── README.md                  # Project documentation
├── README_analysis.md         # Analysis documentation
└── visualizations.ipynb      # Visualizations and Graphs of Comparisons
```

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/SPM587SP24/CS584_ML_Project2.git
cd CS584_ML_Project2
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run tests:
```bash
pytest BoostingTrees/test/test_boosting_trees.py -v -s
```

5. View Analysis and Visualizations:
```bash
jupyter notebook visualizations.ipynb
```
This notebook contains detailed graphs, performance analysis, and comparisons of the model.

## Parameters and Usage

### Key Parameters

1. `n_estimators` (default=100): Number of boosting stages
2. `learning_rate` (default=0.1): Shrinks the contribution of each tree
3. `max_depth` (default=3): Maximum depth of individual trees
4. `min_samples_split` (default=2): Minimum samples required to split a node
5. `early_stopping_rounds` (default=5): Number of rounds without improvement to stop
6. `validation_fraction` (default=0.1): Fraction of training data for validation
7. `n_jobs` (default=-1): Number of parallel jobs (-1 means all processors)
8. `custom_loss` (default=None): Optional custom loss function

## Model Implementation Details

The model implementation (`BoostingTrees.py`) includes:

1. **Core Classes**:
   - `DecisionTree`: Represents individual decision tree nodes
   - `BoostingTreesClassifier`: Main classifier implementation

2. **Key Methods**:
   - `fit`: Trains the model on input data
   - `predict`: Makes class predictions
   - `predict_proba`: Returns class probabilities
   - `get_feature_importance`: Calculates feature importance
   - `plot_learning_curves`: Visualizes training progress
   - `plot_feature_importance`: Visualizes feature importance

3. **Advanced Features**:
   - Parallel tree building
   - Early stopping
   - Custom loss functions
   - Feature importance calculation
   - Learning curve tracking

## Test Suite Details

The test suite (`test_boosting_trees.py`) includes:

1. **Test Categories**:
   - Basic functionality tests
   - Advanced feature tests
   - Edge case tests
   - Performance tests

2. **Test Coverage**:
   - All core functionality
   - All advanced features
   - Edge cases and error handling
   - Performance characteristics

3. **Test Methodology**:
   - Unit tests for individual components
   - Integration tests for full pipeline
   - Performance benchmarks
   - Error handling verification

## Contributing

Feel free to submit issues and enhancement requests!

## Contributors

- Neel Patel (A20524638) - npatel157@hawk.iit.edu
- Karan Savaliya (A20539487) - ksavaliya@hawk.iit.edu
- Deep Patel (A20545631) - dpatel224@hawk.iit.edu
- Johan Vijayan (A20553527) - jvijayan1@hawk.iit.edu

Additional Contributions are welcome! Feel free to submit a pull request with improvements or fixes.

## References

1. **Model Implementation**:
   - Elements of Statistical Learning (2nd Edition), Sections 10.9-10.10
   - Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction
   - Scikit-learn Gradient Boosting Classifier Documentation

2. **Test Cases**:
   - Scikit-learn Test Suite
   - Python Testing with pytest
   - Machine Learning Testing Best Practices

3. **Visualization Notebook**:
   - Matplotlib Documentation
   - Seaborn Statistical Data Visualization
   - Jupyter Notebook Best Practices
   - Scikit-learn Model Evaluation Documentation
   