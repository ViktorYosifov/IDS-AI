from sklearn.ensemble import IsolationForest

def detect_anomalies(X_train, X_test):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(X_train)
    anomalies = iso_forest.predict(X_test)
    anomalies = [1 if x == -1 else 0 for x in anomalies]
    return anomalies
