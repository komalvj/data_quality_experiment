import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA


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


# Function to evaluate the models
def evaluate_models(X, y, model_name, imbalance_level):
    
    print(f"Model: {model_name}, Imbalance level: {imbalance_level}")

    scores = {}

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clusterer', models[model_name])
    ])

    X_imbalanced, y_imbalanced = create_imbalanced_data(X, y, imbalance_level)


    pipeline.fit(X_imbalanced)
    y_pred = pipeline.predict(X_imbalanced) 

    # Calculate ARI score
    ari_score = adjusted_rand_score(y_imbalanced, y_pred)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(pipeline.named_steps['scaler'].transform(X_imbalanced), y_pred)
    
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
            x=results['imbalance_levels'],
            y=results['results'][f'{model_name}_ARI'],
            marker='o',
            color=color_palette[model_index],
            label=f'{model_name}',
            ax=ax1
        )

        sns.lineplot(
            x=results['imbalance_levels'],
            y=results['results'][f'{model_name}_Silhouette'],
            marker='o',
            color=color_palette[model_index],
            ax=ax2
        )   
    
    ax1.set_xticklabels([], fontweight='bold')
    ax1.set_ylabel('Adjusted Rand Index (ARI)', fontweight='bold')
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
    # ax1.set_title('Impact of Class Imbalance on Clustering Models')
    ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)

    ax2.set_xlabel('Imbalance Level', fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontweight='bold')
    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Define imbalance levels and scenarios
imbalance_levels = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]
# scenarios = ['train', 'test', 'train+test']
models = {
    'KMeans': KMeans(n_clusters=2, random_state=27),
    'GaussianMixture': GaussianMixture(n_components=2, random_state=27)
}


# Run experiments
results = {
    'models': models,
    'imbalance_levels': imbalance_levels, 
    'results': {f'{model}_ARI': [] for model in models}
}
results['results'].update({f'{model}_Silhouette': [] for model in models})

for model_name in list(models.keys()):
    for imbalance_level in imbalance_levels:
        scores = evaluate_models(X, y, model_name, imbalance_level)
        results['results'][f'{model_name}_ARI'].append(scores[f'{model_name}_ARI'])
        results['results'][f'{model_name}_Silhouette'].append(scores[f'{model_name}_Silhouette'])

print(results)

# Plot the results
plot_results(results)