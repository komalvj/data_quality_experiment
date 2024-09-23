import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats


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
        ('classifier', models[model_name])
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
        cv_scores.append(accuracy_score(y_test_dup, y_pred))
    
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
    plt.ylabel('Accuracy', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    # plt.title(f'Impact of Data Duplication on Classification Models')
    plt.legend(title='Model - Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Performing quantitative analysis
# def analyze_results(results):
#     for model_name in results['models']:
#         print(f"\nAnalysis for {model_name}")
#         for scenario in results['scenarios']:
#             print(f"\n  Scenario: {scenario}")
#             x = np.array(results['duplication_factors'])
#             y = np.array(results['results'][model_name][scenario])
            
#             # 1. Overall change
#             total_change = y[-1] - y[0]
#             percent_change = (total_change / y[0]) * 100
#             print(f"    Overall change: {total_change:.4f} ({percent_change:.2f}%)")
            
#             # 2. Maximum performance drop
#             max_drop = np.max(y) - np.min(y)
#             max_drop_percent = (max_drop / np.max(y)) * 100
#             print(f"    Maximum performance drop: {max_drop:.4f} ({max_drop_percent:.2f}%)")
            
#             # 3. Trend analysis
#             slope, _, r_value, _, _ = stats.linregress(x, y)
#             print(f"    Trend: slope = {slope:.4f}, R² = {r_value**2:.4f}")


# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Define duplication factors and scenarios
duplication_factors = [0, 0.1, 0.25, 0.5, 0.8, 1.0]
scenarios = ['train', 'test', 'train+test']
models = {
    'SVM': SVC(random_state=27),
    'KNN': KNeighborsClassifier()
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
