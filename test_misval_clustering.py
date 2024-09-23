import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
# from sklearn.cluster import AgglomerativeClustering


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


# Function to evaluate the models
def evaluate_models(X, y, model_name, missing_type, proportion):

    print(f"Model: {model_name}, Missing level: {missing_type}, {proportion}")
    
    scores = {}
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clusterer', models[model_name])
    ])

    X_polluted = introduce_missing_data(X, missing_type, proportion)
    
    pipeline.fit(X_polluted)
    y_pred = pipeline.predict(X_polluted)

    # Calculate ARI score
    ari_score = adjusted_rand_score(y, y_pred)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(pipeline.named_steps['scaler'].transform(
        pipeline.named_steps['imputer'].transform(X_polluted)
    ), y_pred)
    
    scores[f'{model_name}_ARI'] = ari_score
    scores[f'{model_name}_Silhouette'] = silhouette_avg
    
    return scores

# Plotting function using seaborn
def plot_results(results, missing_type):
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
    # ax1.set_title(f'Impact of Missing Data ({missing_type}) on Clustering Models')
    ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    ax2.set_xlabel('Pollution Level', fontweight='bold')
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
# wine = load_wine()
# X, y = wine.data, wine.target

# Define pollution levels and scenarios
pollution_levels = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]
# scenarios = ['train', 'test', 'train+test']
missing_types = ['MCAR', 'MNAR']
models = {
    'KMeans': KMeans(n_clusters=3, random_state=27),
    'GaussianMixture': GaussianMixture(n_components=3, random_state=27),
    # 'AgglomerativeClustering': AgglomerativeClustering(n_clusters=3)
}


# Run experiments for each missing data type
for missing_type in missing_types:
    results = {
        'models': models, 
        'pollution_levels': pollution_levels, 
        'results': {f'{model}_ARI': [] for model in models}
    }
    results['results'].update({f'{model}_Silhouette': [] for model in models})

    for model_name in list(models.keys()):
        for proportion in pollution_levels:
            scores = evaluate_models(X, y, model_name, missing_type, proportion)
            results['results'][f'{model_name}_ARI'].append(scores[f'{model_name}_ARI'])
            results['results'][f'{model_name}_Silhouette'].append(scores[f'{model_name}_Silhouette'])

    print(results)

    # Plot the results for each missing data type
    plot_results(results, missing_type)
