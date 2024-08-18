import numpy as np
import pandas as pd
from data_preparation import load_and_split_data
from data_pollution import add_noise, reduce_feature_accuracy, reduce_target_accuracy, reduce_uniqueness, introduce_inconsistent_representation
from model_training import get_classification_models, get_regression_models
from evaluation import evaluate_classification_model, evaluate_regression_model
from visualization import plot_performance

def run_experiment():
    # Load and split data
    (X_train_class, X_test_class, y_train_class, y_test_class,
     X_train_reg, X_test_reg, y_train_reg, y_test_reg) = load_and_split_data()
    
    # Define models
    classification_models = get_classification_models()
    regression_models = get_regression_models()

    # Define pollution levels
    pollution_levels = np.linspace(0, 0.5, 6)  # From 0% to 50%
    
    # Initialize performance dictionaries
    class_performance_train_polluted = {model: [] for model in classification_models}
    reg_performance_train_polluted = {model: [] for model in regression_models}
    
    # Experiment for training data polluted
    for pollution in pollution_levels:
        X_train_class_polluted = add_noise(X_train_class, noise_level=pollution)
        X_train_reg_polluted = add_noise(X_train_reg, noise_level=pollution)

        # Classification
        for model_name, model in classification_models.items():
            accuracy, report = evaluate_classification_model(model, X_train_class_polluted, y_train_class, X_test_class, y_test_class)
            class_performance_train_polluted[model_name].append(accuracy)
        
        # Regression
        for model_name, model in regression_models.items():
            mse, r2 = evaluate_regression_model(model, X_train_reg_polluted, y_train_reg, X_test_reg, y_test_reg)
            reg_performance_train_polluted[model_name].append(mse)
    
    # Plot results
    for model_name in classification_models:
        plot_performance(pollution_levels, class_performance_train_polluted, [model_name], 'Accuracy', f'Classification Model Performance (Training Data Polluted) - {model_name}')
    
    for model_name in regression_models:
        plot_performance(pollution_levels, reg_performance_train_polluted, [model_name], 'Mean Squared Error', f'Regression Model Performance (Training Data Polluted) - {model_name}')

if __name__ == "__main__":
    run_experiment()
