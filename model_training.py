from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor

def get_classification_models():
    return {
        'Support Vector Machine': SVC(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'k-Nearest Neighbors': KNeighborsClassifier()
    }

def get_regression_models():
    return {
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge()
    }
