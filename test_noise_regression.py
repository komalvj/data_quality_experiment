import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
# from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Function to introduce noise to the dataset
def introduce_noise(X, y, noise_level, type):
    np.random.seed(27)
    X_noise = np.random.normal(0, noise_level, X.shape)
    y_noise = np.random.normal(0, noise_level, y.shape)

    if type == "feature":
        return X + X_noise, y
    elif type == "target":
        return X, y + y_noise
    else:
        return X + X_noise, y + y_noise
    

# Function to evaluate the models with cross-validation
def evaluate_models(X, y, model_name, inaccuracy_type, noise_level, scenario, cv_folds=5):
    
    print(f"Model: {model_name}, Noise level: {inaccuracy_type}, {noise_level}, Scenario: {scenario}")

    scores = {}
    cv_scores = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=27)

    pipeline = Pipeline([
        # ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regressor', models[model_name])
    ])
     
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if scenario == 'train':
            X_train_noisy, y_train_noisy = introduce_noise(X_train, y_train, noise_level, inaccuracy_type)
            X_test_noisy, y_test_noisy = X_test, y_test
        elif scenario == 'test':
            X_train_noisy, y_train_noisy = X_train, y_train
            X_test_noisy, y_test_noisy = introduce_noise(X_test, y_test, noise_level, inaccuracy_type)
        elif scenario == 'train+test':
            X_train_noisy, y_train_noisy = introduce_noise(X_train, y_train, noise_level, inaccuracy_type)
            X_test_noisy, y_test_noisy = introduce_noise(X_test, y_test, noise_level, inaccuracy_type)
        
        pipeline.fit(X_train_noisy, y_train_noisy)
        y_pred = pipeline.predict(X_test_noisy)
        cv_scores.append(mean_squared_error(y_test_noisy, y_pred))
        
    scores[model_name] = np.mean(cv_scores)
    
    return scores


# Plotting function using seaborn
def plot_results(results, inaccuracy_type):
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
                x=results['noise_levels'],
                y=results['results'][model_name][scenario],
                marker='o',
                linestyle=line_styles[scenario],
                color=color_palette[model_index],
                label=f'{model_name} - {scenario}'
            )


    # Uncomment for diabetes dataset
    # if inaccuracy_type != 'target':        
    #     plt.yscale('log') # Logarithmic scale
    #     yticks = [3000, 3500, 4000, 5000, 6000, 10000, 50000, 100000, 500000, 1000000]
    #     plt.yticks(yticks, [f'{y:,}' for y in yticks])

    plt.xlabel('Noise Level', fontweight='bold')
    plt.ylabel('Mean Squared Error (MSE)', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    # plt.title(f'Impact of Noise ({inaccuracy_type}) on Regression Models')
    plt.legend(title='Model - Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Validation
iris = load_iris()
X, y = iris.data, iris.target

# Load dataset
# diabetes = load_diabetes()
# X, y = diabetes.data, diabetes.target

# Define noise levels and scenarios
noise_levels = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8]
scenarios = ['train', 'test', 'train+test']
inaccuracy_types = ['feature', 'target', 'feature+target']
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=27),
    'RidgeRegression': Ridge(alpha=0.15, random_state=27)
}

# Run experiments for each inaccuracy data type
for inaccuracy_type in inaccuracy_types:
    results = {'models': models, 'scenarios': scenarios, 'noise_levels': noise_levels, 'results': {model: {scenario: [] for scenario in scenarios} for model in models}}

    for model_name in list(models.keys()):
        for scenario in scenarios:
            for noise_level in noise_levels:
                scores = evaluate_models(X, y, model_name, inaccuracy_type, noise_level, scenario)
                results['results'][model_name][scenario].append(scores[model_name])

    print(results)
    
    # Plot the results
    plot_results(results, inaccuracy_type)