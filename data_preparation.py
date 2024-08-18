import pandas as pd
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

def load_and_split_data():
    # Load datasets
    iris = load_iris()
    boston = load_boston()

    # Create DataFrames
    X_classification = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_classification = pd.Series(iris.target)

    X_regression = pd.DataFrame(boston.data, columns=boston.feature_names)
    y_regression = pd.Series(boston.target)

    # Split datasets
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_classification, y_classification, test_size=0.3, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_regression, y_regression, test_size=0.3, random_state=42)

    return (X_train_class, X_test_class, y_train_class, y_test_class,
            X_train_reg, X_test_reg, y_train_reg, y_test_reg)
