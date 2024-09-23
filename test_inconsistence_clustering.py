import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
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
            if col_index == 2:  # Example: person_home_ownership column
                X[idx, col_index] = np.random.choice(['rent', 'Rent'])
            # You can add more inconsistencies for other categorical columns here if needed
    return X


# Function to evaluate the models
def evaluate_models(X, y, model_name, proportion):

    print(f"Model: {model_name}, Inconsistency level: {proportion}")
    
    scores = {}

    categorical_indices = [2, 4, 5, 9]  # categorical column indices
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

    X_inconsistent = introduce_inconsistent_data(X, proportion, categorical_indices)
    
    pipeline.fit(X_inconsistent)
    y_pred = pipeline.predict(X_inconsistent)
    
    # Calculate ARI score
    ari_score = adjusted_rand_score(y, y_pred)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(pipeline.named_steps['preprocessor'].transform(X_inconsistent), y_pred)
    
    scores[f'{model_name}_ARI'] = ari_score
    scores[f'{model_name}_Silhouette'] = silhouette_avg
    
    return scores


# Plotting function using seaborn
def plot_results(results):
    sns.set(style="whitegrid")
    color_palette = sns.color_palette("tab10", len(results['models']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    for model_index, model_name in enumerate(results['models']):
        sns.lineplot(
            x=results['pollution_levels'],
            y=results['results'][f'{model_name}_ARI'],
            marker='o',
            color=color_palette[model_index],
            label=f'{model_name}', 
            ax=ax1
        )

        sns.lineplot(
            x=results['pollution_levels'],
            y=results['results'][f'{model_name}_Silhouette'],
            marker='o',
            color=color_palette[model_index],
            ax=ax2
        )
    
    ax1.set_xticklabels([], fontweight='bold')
    ax1.set_ylabel('Adjusted Rand Index (ARI)', fontweight='bold')
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
    # ax1.set_title('Impact of Inconsistently represented data on Clustering Models')
    ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    ax2.set_xlabel('Inconsistency Factor', fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontweight='bold')
    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    # plt.savefig()


# Load dataset
cr_loan = pd.read_csv('cr_loan_clean.csv')
X = cr_loan.drop(columns=['loan_status']).values
y = cr_loan['loan_status'].values

# Define pollution levels and scenarios
pollution_levels = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]
# scenarios = ['train', 'train+test']
models = {
    'KMeans': KMeans(n_clusters=2, random_state=27),
    'GaussianMixture': GaussianMixture(n_components=2, random_state=27)
}


# Run experiments for each missing data type
results = {
    'models': models, 
    'pollution_levels': pollution_levels, 
    'results': {f'{model}_ARI': [] for model in models}
}
results['results'].update({f'{model}_Silhouette': [] for model in models})

for model_name in list(models.keys()):
    for proportion in pollution_levels:
        scores = evaluate_models(X, y, model_name, proportion)
        results['results'][f'{model_name}_ARI'].append(scores[f'{model_name}_ARI'])
        results['results'][f'{model_name}_Silhouette'].append(scores[f'{model_name}_Silhouette'])

print(results)

# Plot the results for each missing data type
plot_results(results)