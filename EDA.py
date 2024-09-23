import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import mutual_info_regression

def load_and_prepare_data():
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name='target')
    return pd.concat([X, y], axis=1)

def create_summary_table(df):
    summary = df.describe().T
    summary['skew'] = df.skew()
    summary['kurtosis'] = df.kurtosis()
    return summary.round(3)

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    # plt.title("Correlation Heatmap", fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    # plt.savefig('diabetes_corr.png')
    plt.show()

def plot_histogram(data, title, xlabel, ylabel="Frequency"):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor='black')
    # plt.title(title, fontweight='bold')
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    # plt.savefig('diabetes_target_univeriate_analysis.png')
    plt.show()

def calculate_mutual_info(X, y):
    mi_scores = mutual_info_regression(X, y)
    return pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

def plot_mutual_info(mi_scores):
    plt.figure(figsize=(10, 6))
    plt.barh(mi_scores.index, mi_scores.values, color='skyblue')
    plt.xlabel('Mutual Information Score', fontweight='bold')
    # plt.title('Mutual Information Regression Scores for Features', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.savefig('diabetes_mutual_info_regression_scores.png')
    plt.show()

def plot_top_features(df, features):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    # fig.suptitle("Distribution of Top 3 Important Features", fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(features):
        axes[i].hist(df[feature], bins=30, edgecolor='black')
        axes[i].set_title(f"Distribution of {feature}", fontweight='bold')
        axes[i].set_xlabel(feature, fontweight='bold')
        axes[0].set_ylabel("Count", fontweight='bold')
    
    plt.tight_layout()
    # plt.savefig('diabetes_top3.png')
    plt.show()

def main():
    df = load_and_prepare_data()
    
    summary_table = create_summary_table(df)
    print("Summary Statistics for All Features:")
    print(summary_table.to_string())
    
    plot_correlation_heatmap(df)
    plot_histogram(df['target'], "Distribution of Target", "Target")
    
    X = df.drop('target', axis=1)
    y = df['target']
    mi_scores = calculate_mutual_info(X, y)
    print("\nMutual Information Scores:")
    print(mi_scores)
    
    plot_mutual_info(mi_scores)
    
    important_features = mi_scores.nlargest(3).index.tolist()
    print("\nTop 3 important features based on mutual information:")
    print(important_features)
    
    plot_top_features(df, important_features)

if __name__ == "__main__":
    main()