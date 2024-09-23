
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Results to be manually inserted 
# Dataset 1
dataset1 = {'models': {'RandomForest': RandomForestRegressor(random_state=27), 'RidgeRegression': Ridge(alpha=0.15, random_state=27)}, 'scenarios': ['train', 'test', 'train+test'], 'noise_levels': [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8], 'results': {'RandomForest': {'train': [3534.7460844356488, 3268.353308968335, 3464.226646386619, 4121.591401973952, 5569.672454803371, 5816.995881670072, 5906.8067322293155], 'test': [3534.7460844356488, 3395.889293748724, 4543.202942688968, 6163.165370617977, 8011.860367706843, 8513.384918878957, 8832.63860915475], 'train+test': [3534.7460844356488, 3212.3502314657817, 4363.808301251277, 5225.52796038049, 6415.412883455056, 6887.091804749744, 7275.625375117466]}, 'RidgeRegression': {'train': [3009.3265445796696, 3021.0333346988223, 3457.6988647950143, 4341.947924366848, 5527.427142854202, 5868.09520440296, 5944.855301105095], 'test': [3009.3265445796696, 3159.4931156241064, 8448.371227357298, 25607.945479023543, 147409.84564735397, 584823.6039723436, 1496497.5897356553], 'train+test': [3009.3265445796696, 2991.509148727841, 4027.9373869485453, 5161.301965998593, 6044.781426484444, 6227.580513674931, 6267.5986458341695]}}}

# Dataset 2
dataset2 = {'models': {'RandomForest': RandomForestRegressor(random_state=27), 'RidgeRegression': Ridge(alpha=0.15, random_state=27)}, 'scenarios': ['train', 'test', 'train+test'], 'noise_levels': [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8], 'results': {'RandomForest': {'train': [0.03228333333333333, 0.03135266666666667, 0.034999333333333334, 0.03328666666666666, 0.04162, 0.06010333333333334, 0.09551733333333334], 'test': [0.03228333333333333, 0.03357533333333333, 0.039598, 0.04500333333333333, 0.04738866666666667, 0.11760066666666666, 0.19504133333333334], 'train+test': [0.03228333333333333, 0.03193733333333333, 0.03444533333333333, 0.03753066666666667, 0.06649666666666668, 0.10184666666666667, 0.17308266666666666]}, 'RidgeRegression': {'train': [0.048162217323393375, 0.04813500524061232, 0.04806311441083647, 0.048560654046176675, 0.054923132247323994, 0.06589685854544186, 0.07717256966752507], 'test': [0.048162217323393375, 0.04848712609436679, 0.05054257951420976, 0.054812487544899244, 0.07895948667620627, 0.1569954020258466, 0.3129935131612272], 'train+test': [0.048162217323393375, 0.04845915771898306, 0.050208575369194194, 0.0532139692606813, 0.0651758263980086, 0.09245638429375133, 0.13658532205097054]}}}

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def analyze_performance_model_wise(dataset1, dataset2):
    noise_levels = np.array(dataset1['noise_levels'])
    models = list(dataset1['models'].keys())
    scenarios = dataset1['scenarios']
    # scenarios = ['ARI', 'Silhouette']
    
    fig, axs = plt.subplots(len(models), len(scenarios), figsize=(17, 10))
    # fig.suptitle(f"Model Performance Comparison Across Datasets", fontsize=16)
    
    for i, model in enumerate(models):
        for j, scenario in enumerate(scenarios):

            # Regression # MSE results need to be normalized due to difference in scale
            performances1 = scaler.fit_transform(np.array(dataset1['results'][model][scenario]).reshape(-1, 1)).flatten()    
            performances2 = scaler.fit_transform(np.array(dataset2['results'][model][scenario]).reshape(-1, 1)).flatten()

            # Clustering
            # performances1 = np.array(dataset1['results'][f'{model}_{scenario}']) 
            # performances2 = np.array(dataset2['results'][f'{model}_{scenario}'])

            # performances1 = np.array(dataset1['results'][model][scenario])
            # performances2 = np.array(dataset2['results'][model][scenario])

                                                 
            ax = axs[i, j]
            
            # Plot Dataset 1
            ax.scatter(noise_levels, performances1, color='mediumturquoise', label='Diabetes')
            slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(noise_levels, performances1)
            ax.plot(noise_levels, intercept1 + slope1 * noise_levels, color='mediumturquoise', linestyle='--')
            corr1, corr_p1 = stats.pearsonr(noise_levels, performances1)
            
            # Plot Dataset 2
            ax.scatter(noise_levels, performances2, color='orangered', label='Iris')
            slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(noise_levels, performances2)
            ax.plot(noise_levels, intercept2 + slope2 * noise_levels, color='orangered', linestyle='--')
            corr2, corr_p2 = stats.pearsonr(noise_levels, performances2)
            
            ax.set_title(f"{model} - {scenario}", fontsize=9, fontweight='bold')
            ax.legend()
            
            # Set y-label only for the first column
            if j == 0:
                ax.set_ylabel("Performance", fontsize=9)
            
            # Set x-label only for the last row
            if i == len(models) - 1:
                ax.set_xlabel("Noise Level", fontsize=9)
            
            # Add statistical information to the plot with smaller font
            ax.text(0.05, 0.05, 
                    f'Diabetes: Slope={slope1:.2f}, p={p_value1:.2f}, Corr={corr1:.2f}\n'
                    f'Iris: Slope={slope2:.2f}, p={p_value2:.2f}, Corr={corr2:.2f}', 
                    transform=ax.transAxes, fontsize=8)

    

    plt.tight_layout()
    return fig

# Analyze both datasets
model_wise_plot = analyze_performance_model_wise(dataset1, dataset2)

plt.show()  # This will display the plot