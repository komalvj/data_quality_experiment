import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


# Function to add class imbalance to the dataset
def create_imbalanced_data(X, y, minority_fraction):  
    """
    Introduces class imbalance into the dataset.

    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Target vector.
    minority_fraction (float): The desired fraction of the minority class in the dataset.

    Returns:
    numpy.ndarray: Feature matrix with class imbalance.
    numpy.ndarray: Target vector with class imbalance.
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    majority_class = unique_classes[np.argmax(class_counts)]
    minority_class = unique_classes[np.argmin(class_counts)]

    X_majority = X[y == majority_class]
    y_majority = y[y == majority_class]
    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]
    
    minority_size = int((1 - minority_fraction) * len(X_minority))
    X_minority_undersampled = X_minority[:minority_size]
    y_minority_undersampled = y_minority[:minority_size]
    
    X_imbalanced = np.vstack((X_majority, X_minority_undersampled))
    y_imbalanced = np.hstack((y_majority, y_minority_undersampled))
    
    return X_imbalanced, y_imbalanced


# Function to evaluate the models with cross-validation
def evaluate_models(X, y, model_name, imbalance_level, scenario, cv_folds=5):
    
    print(f"Model: {model_name}, Imbalance level: {imbalance_level}, Scenario: {scenario}")

    scores = {}
    cv_scores = []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=27)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', models[model_name])
    ])

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if scenario == "train":
            X_train_imbalanced, y_train_imbalanced = create_imbalanced_data(X_train, y_train, imbalance_level)
            X_test_imbalanced, y_test_imbalanced = X_test, y_test
        elif scenario == "test":
            X_train_imbalanced, y_train_imbalanced = X_train, y_train
            X_test_imbalanced, y_test_imbalanced = create_imbalanced_data(X_test, y_test, imbalance_level)
        elif scenario == "train+test":
            X_train_imbalanced, y_train_imbalanced = create_imbalanced_data(X_train, y_train, imbalance_level)
            X_test_imbalanced, y_test_imbalanced = create_imbalanced_data(X_test, y_test, imbalance_level)            
        
        pipeline.fit(X_train_imbalanced, y_train_imbalanced)
        y_pred = pipeline.predict(X_test_imbalanced)

        cv_scores.append(f1_score(y_test_imbalanced, y_pred, average='weighted'))
            
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
                x=results['imbalance_levels'],
                y=results['results'][model_name][scenario],
                marker='o',
                linestyle=line_styles[scenario],
                color=color_palette[model_index],
                label=f'{model_name} - {scenario}'
            )
        
    plt.xlabel('Imbalance Level', fontweight='bold')
    plt.ylabel('Weighted F1-score', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    # plt.title('Impact of Class Imbalance on Classification Models')
    plt.legend(title='Model - Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Define imbalance levels and scenarios
imbalance_levels = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]
scenarios = ['train', 'test', 'train+test']
models = {
    'SVM': SVC(random_state=27),
    'KNN': KNeighborsClassifier()
}


# Run experiments
results = {'models': models, 'scenarios': scenarios, 'imbalance_levels': imbalance_levels, 'results': {model: {scenario: [] for scenario in scenarios} for model in models}}

for model_name in list(models.keys()):
    for scenario in scenarios:
        for imbalance_level in imbalance_levels:
            scores = evaluate_models(X, y, model_name, imbalance_level, scenario)
            results['results'][model_name][scenario].append(scores[model_name])


print(results)

# Plot the results
plot_results(results)