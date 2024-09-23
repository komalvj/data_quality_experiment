import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
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


# Function to evaluate the models
def evaluate_models(X, y, model_name, duplication_factor):
    print(f"Model: {model_name}, Duplication factor: {duplication_factor}")

    scores = {}
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clusterer', models[model_name])
    ])

    X_dup, y_dup = add_duplicates(X, y, duplication_factor)
    
    pipeline.fit(X_dup)
    y_pred = pipeline.predict(X_dup)
    
    # Calculate ARI score
    ari_score = adjusted_rand_score(y_dup, y_pred)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(pipeline.named_steps['scaler'].transform(X_dup), y_pred)
    
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
            x=results['duplication_factors'],
            y=results['results'][f'{model_name}_ARI'],
            marker='o',
            color=color_palette[model_index],
            label=f'{model_name}',
            ax=ax1
        )

        sns.lineplot(
            x=results['duplication_factors'],
            y=results['results'][f'{model_name}_Silhouette'],
            marker='o',
            color=color_palette[model_index],
            ax=ax2
        )
    
    ax1.set_xticklabels([], fontweight='bold')
    ax1.set_ylabel('Adjusted Rand Index (ARI)', fontweight='bold')
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
    # ax1.set_title('Impact of Data Duplication on Clustering Models')
    ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    ax2.set_xlabel('Duplication Factor', fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontweight='bold')
    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
            

# Validation
iris = load_iris()
X, y = iris.data, iris.target

# Load dataset
# housing = fetch_california_housing()
# X, y = housing.data, housing.target

# Define duplication factors and models
duplication_factors = [0, 0.1, 0.25, 0.5, 0.8, 1.0]
models = {
    'KMeans': KMeans(n_clusters=5, random_state=27),
    'GaussianMixture': GaussianMixture(n_components=5, random_state=27)
}

# Run experiments
results = {
    'models': models, 
    'duplication_factors': duplication_factors, 
    'results': {f'{model}_ARI': [] for model in models}
}
results['results'].update({f'{model}_Silhouette': [] for model in models})

for model_name in list(models.keys()):
    for duplication_factor in duplication_factors:
        scores = evaluate_models(X, y, model_name, duplication_factor)
        results['results'][f'{model_name}_ARI'].append(scores[f'{model_name}_ARI'])
        results['results'][f'{model_name}_Silhouette'].append(scores[f'{model_name}_Silhouette'])

print(results)

# Plot the results
plot_results(results)