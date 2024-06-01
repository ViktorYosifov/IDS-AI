from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt', 'log2']
    }