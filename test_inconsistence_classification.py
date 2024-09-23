import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Function to introduce inconsistent representation in data
def introduce_inconsistent_data(X, inconsistency_factor=0.1, cat_column_indices=None):
    """
    Introduces inconsistent representation into the dataset.

    Parameters:
    X (numpy.ndarray): Feature matrix.
    inconsistency_factor (float): Proportion of entries to introduce inconsistencies.
    cat_column_indices (list): List of column indices of categorical features.

    Returns:
    numpy.ndarray: Feature matrix with inconsistent representation.
    """
    np.random.seed(27)
    num_inconsistencies = int(X.shape[0] * inconsistency_factor)
    for col_index in cat_column_indices:
        inconsistent_indices = np.random.choice(X.shape[0], num_inconsistencies, replace=True)
        for idx in inconsistent_indices:
            if col_index == 2:  # person_home_ownership column
                X[idx, col_index] = np.random.choice(['rent', 'Rent']) # original value = 'RENT'
    return X


# Function to evaluate models with cross-validation
def evaluate_models(X, y, model_name, proportion, scenario, cv_folds=5):

    print(f"Model: {model_name}, Inconsistency level: {proportion}, Scenario: {scenario}")
    
    scores = {}
    cv_scores = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=27)

    categorical_indices = [2, 4, 5, 9]
    numeric_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 
                Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), 
                                ('scaler', StandardScaler())]
                        ), numeric_indices),
            ('cat', OneHotEncoder(drop='first'), categorical_indices)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clusterer', models[model_name])
    ])
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if scenario == 'train':
            X_train_p = introduce_inconsistent_data(X_train, proportion, categorical_indices)
            X_test_p = X_test
        elif scenario == 'test':
            X_train_p = X_train
            X_test_p = introduce_inconsistent_data(X_test, proportion, categorical_indices)
        elif scenario == 'train+test':
            X_train_p = introduce_inconsistent_data(X_train, proportion, categorical_indices)
            X_test_p = introduce_inconsistent_data(X_test, proportion, categorical_indices)            
        
        pipeline.fit(X_train_p, y_train)
        y_pred = pipeline.predict(X_test_p)
        cv_scores.append(accuracy_score(y_test, y_pred))
        
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

    fig = plt.figure(figsize=(12, 8))

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

    plt.xlabel('Inconsistency Factor', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    # plt.title(f'Impact of Inconsistently represented data on Classification Models')
    plt.legend(title='Model - Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.savefig('temp.png', dpi=fig.dpi)

# Load dataset
cr_loan = pd.read_csv('cr_loan_clean.csv')
X = cr_loan.drop(columns=['loan_status']).values
y = cr_loan['loan_status'].values

# Define pollution levels and scenarios
pollution_levels = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]
scenarios = ['train', 'train+test']
models = {
    'SVM': SVC(random_state=27),
    'KNN': KNeighborsClassifier()
}


# Run experiments for each missing data type
results = {'models': models, 'scenarios': scenarios, 'pollution_levels': pollution_levels, 'results': {model: {scenario: [] for scenario in scenarios} for model in models}}

for model_name in list(models.keys()):
    for scenario in scenarios:
        for proportion in pollution_levels:
            scores = evaluate_models(X, y, model_name, proportion, scenario)
            results['results'][model_name][scenario].append(scores[model_name])

print(results)

# Plot the results for each missing data type
plot_results(results)