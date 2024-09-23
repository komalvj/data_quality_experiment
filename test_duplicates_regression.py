import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Function to add duplicates to the dataset
def add_duplicates(X, y, duplication_factor):
    """
    Introduces duplicate data into the dataset.

    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Target vector.
    duplication_factor (float): The proportion of data to be duplicated.

    Returns:
    numpy.ndarray: Feature matrix with duplicates.
    numpy.ndarray: Target vector with duplicates.
    """
    np.random.seed(27)
    num_duplicates = int(X.shape[0] * duplication_factor)
    duplicate_indices = np.random.choice(X.shape[0], num_duplicates, replace=True)
    X_duplicates = np.vstack((X, X[duplicate_indices]))
    y_duplicates = np.hstack((y, y[duplicate_indices]))
    return X_duplicates, y_duplicates


# Function to evaluate the models with cross-validation
def evaluate_models(X, y, model_name, duplication_factor, scenario, cv_folds=5): 

    print(f"Model: {model_name}, Duplication factor: {duplication_factor}, Scenario: {scenario}")

    scores = {}
    cv_scores = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=27)
         
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', models[model_name])
    ])

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if scenario == 'train':
            X_train_dup, y_train_dup = add_duplicates(X_train, y_train, duplication_factor)
            X_test_dup, y_test_dup = X_test, y_test
        elif scenario == 'test':
            X_train_dup, y_train_dup = X_train, y_train
            X_test_dup, y_test_dup = add_duplicates(X_test, y_test, duplication_factor)
        elif scenario == 'train+test':
            X_train_dup, y_train_dup = add_duplicates(X_train, y_train, duplication_factor)
            X_test_dup, y_test_dup = add_duplicates(X_test, y_test, duplication_factor)

        pipeline.fit(X_train_dup, y_train_dup)
        y_pred = pipeline.predict(X_test_dup)
        cv_scores.append(mean_squared_error(y_test_dup, y_pred))
    
    scores[model_name] = np.mean(cv_scores)
    
    return scores


# Plotting function using seaborn
def plot_results(results):
    sns.set(style="whitegrid")
    line_styles = {
        'train': 'solid',
        'test': 'dotted',
        'train+test': 'dashdot'
    }  
    color_palette = sns.color_palette("tab10", len(results['models']))

    plt.figure(figsize=(12, 8))

    for model_index, model_name in enumerate(results['models']):
        for scenario in results['scenarios']:
            sns.lineplot(
                x=results['duplication_factors'],
                y=results['results'][model_name][scenario],
                marker='o',
                linestyle=line_styles[scenario],
                color=color_palette[model_index],
                label=f'{model_name} - {scenario}'
            )
        
    plt.xlabel('Duplication Factor', fontweight='bold')
    plt.ylabel('Mean Squared Error (MSE)', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    # plt.title('Impact of Data Duplication on Regression Models')
    plt.legend(title='Model - Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Define duplication factors and scenarios
duplication_factors = [0, 0.1, 0.25, 0.5, 0.8, 1.0]
scenarios = ['train', 'test', 'train+test']
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=27),
    'RidgeRegression': Ridge(alpha=0.15, random_state=27)
}


# Run experiments
results = {'models': models, 'scenarios': scenarios, 'duplication_factors': duplication_factors, 'results': {model: {scenario: [] for scenario in scenarios} for model in models}}

for model_name in list(models.keys()):
    for scenario in scenarios:
        for duplication_factor in duplication_factors:
            scores = evaluate_models(X, y, model_name, duplication_factor, scenario)
            results['results'][model_name][scenario].append(scores[model_name])

print(results)

# Plot the results
plot_results(results)