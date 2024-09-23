import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Function to introduce missing data
def introduce_missing_data(X, missing_type, proportion=0.1):
    """
    Introduces missing values into the dataset.

    Parameters:
    X (numpy.ndarray): Feature matrix.
    missing_type (str): Type of missing data ('MCAR' or 'MNAR').
    proportion (float): Proportion of missing values to introduce.

    Returns:
    numpy.ndarray: Feature matrix with missing values.
    """
    np.random.seed(27)
    X_missing = X.copy()
    n_samples, n_features = X.shape
    if missing_type == 'MCAR':
        mask = np.random.rand(n_samples, n_features) < proportion
    # elif missing_type == 'MAR':
    #     mask = np.zeros((n_samples, n_features), dtype=bool)
    #     for i in range(1, n_features):
    #         mask[:, i] = np.random.rand(n_samples) < (proportion * (1 + np.abs(X[:, i-1] - X[:, i-1].mean())))
    elif missing_type == 'MNAR':
        mask = np.zeros_like(X, dtype=bool)
        for i in range(n_features):
            mask[:, i] = np.random.rand(n_samples) < (proportion * (X[:, i] > X[:, i].mean()))
    X_missing[mask] = np.nan
    return X_missing


# Function to evaluate models with cross-validation
def evaluate_models(X, y, model_name, missing_type, proportion, scenario, cv_folds=5):
    
    print(f"Model: {model_name}, Missing level: {missing_type}, {proportion}, Scenario: {scenario}")

    scores = {}
    cv_scores = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=27)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', models[model_name])
    ]) 
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if scenario == 'train':
            X_train_p = introduce_missing_data(X_train, missing_type, proportion)
            X_test_p = X_test
        elif scenario == 'test':
            X_train_p = X_train
            X_test_p = introduce_missing_data(X_test, missing_type, proportion)
        elif scenario == 'train+test':
            X_train_p = introduce_missing_data(X_train, missing_type, proportion)
            X_test_p = introduce_missing_data(X_test, missing_type, proportion)            
        
        pipeline.fit(X_train_p, y_train)
        y_pred = pipeline.predict(X_test_p)
        cv_scores.append(accuracy_score(y_test, y_pred))
        
    scores[model_name] = np.mean(cv_scores)
    
    return scores


# Plotting function using seaborn
def plot_results(results, missing_type):
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
                x=results['pollution_levels'],
                y=results['results'][model_name][scenario],
                marker='o',
                linestyle=line_styles[scenario],
                color=color_palette[model_index],
                label=f'{model_name} - {scenario}'
            )

    plt.xlabel('Pollution Level', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    # plt.title(f'Impact of Missing Data ({missing_type}) on Classification Models')
    plt.legend(title='Model - Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Validation
iris = load_iris()
X, y = iris.data, iris.target

# Load dataset
# wine = load_wine()
# X, y = wine.data, wine.target

# Define pollution levels and scenarios
pollution_levels = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]
scenarios = ['train', 'test', 'train+test']
missing_types = ['MCAR', 'MNAR']
models = {
    # 'LogisticRegression': LogisticRegression(max_iter=10000),
    'SVM': SVC(random_state=27),
    'KNN': KNeighborsClassifier()
}


# Run experiments for each missing data type
for missing_type in missing_types:
    results = {'models': models, 'scenarios': scenarios, 'pollution_levels': pollution_levels, 'results': {model: {scenario: [] for scenario in scenarios} for model in models}}
    
    for model_name in list(models.keys()):
        for scenario in scenarios:
            for proportion in pollution_levels:
                scores = evaluate_models(X, y, model_name, missing_type, proportion, scenario)
                results['results'][model_name][scenario].append(scores[model_name])
    
    print(results)

    # Plot the results for each missing data type
    plot_results(results, missing_type)
