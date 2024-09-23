import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Function to introduce noise to the dataset
def introduce_noise(X, y, noise_level, type):
    np.random.seed(42)
    X_noise = np.random.normal(0, noise_level, X.shape)

    unique_y = np.unique(y)
    n_classes = len(unique_y)
    y_noise = np.random.choice(unique_y, size=y.shape, p=[1/n_classes]*n_classes)
    y_noise_mask = np.random.random(y.shape) < noise_level
    y_noisy = np.where(y_noise_mask, y_noise, y)

    if type == "feature":
        return X + X_noise, y
    elif type == "target":
        return X, y_noisy
    else:
        return X + X_noise, y_noisy
    

# Function to evaluate the models with cross-validation
def evaluate_models(X, y, model_name, inaccuracy_type, noise_level):
    
    print(f"Model: {model_name}, Noise level: {inaccuracy_type}, {noise_level}")

    scores = {}

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clusterer', models[model_name])
    ])

    X_noisy, y_noisy = introduce_noise(X, y, noise_level, inaccuracy_type)
    
    pipeline.fit(X_noisy)
    y_pred = pipeline.predict(X_noisy) 

    # Calculate ARI score
    ari_score = adjusted_rand_score(y_noisy, y_pred)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(pipeline.named_steps['scaler'].transform(X_noisy), y_pred)
    
    scores[f'{model_name}_ARI'] = ari_score
    scores[f'{model_name}_Silhouette'] = silhouette_avg
    
    return scores


# Plotting function using seaborn
def plot_results(results, inaccuracy_type):
    sns.set(style="whitegrid")
    color_palette = sns.color_palette("tab10", len(results['models']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    for model_index, model_name in enumerate(results['models']):
        sns.lineplot(
            x=results['noise_levels'],
            y=results['results'][f'{model_name}_ARI'],
            marker='o',
            color=color_palette[model_index],
            label=f'{model_name}',
            ax=ax1
        )

        sns.lineplot(
            x=results['noise_levels'],
            y=results['results'][f'{model_name}_Silhouette'],
            marker='o',
            color=color_palette[model_index],
            ax=ax2
        )
    
    ax1.set_xticklabels([], fontweight='bold')
    ax1.set_ylabel('Adjusted Rand Index (ARI)', fontweight='bold')
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
    # ax1.set_title(f'Impact of Noise ({inaccuracy_type}) on Clustering Models')
    ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    ax2.set_xlabel('Noise Level', fontweight='bold')
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

# Define noise levels and scenarios
noise_levels = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]
scenarios = ['train', 'test', 'train+test']
inaccuracy_types = ['feature', 'target', 'feature+target']
models = {
    'KMeans': KMeans(n_clusters=3, random_state=42),
    'GaussianMixture': GaussianMixture(n_components=3, random_state=42)
}


# Run experiments for each inaccuracy data type
for inaccuracy_type in inaccuracy_types:
    results = {
        'models': models, 
        'scenarios': scenarios, 
        'noise_levels': noise_levels, 
        'results': {f'{model}_ARI': [] for model in models}
    }
    results['results'].update({f'{model}_Silhouette': [] for model in models})

    for model_name in list(models.keys()):
        for noise_level in noise_levels:
            scores = evaluate_models(X, y, model_name,inaccuracy_type, noise_level)
            results['results'][f'{model_name}_ARI'].append(scores[f'{model_name}_ARI'])
            results['results'][f'{model_name}_Silhouette'].append(scores[f'{model_name}_Silhouette'])

    print(results)

    # Plot the results
    plot_results(results, inaccuracy_type)