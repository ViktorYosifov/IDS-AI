from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model, best_params

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report, y_pred

