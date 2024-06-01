from data_processing import load_data, encode_features, preprocessing_data
from feature_engineering import reduce_dimensionality
from model_training import train_model, evaluate_model
from anomaly_detection import detect_anomalies
from visualization import plot_confusion_matrix, plot_anomalies
from utils import save_model
from sklearn.model_selection import train_test_split

#Load and preprocess data
url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"

column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label'
]

categorical_cols = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

df = load_data(url, column_names)
df = encode_features(df, categorical_cols)
X_scaled, y, scaler = preprocessing_data(df)
X_reduced, pca = reduce_dimensionality(X_scaled)

#Split data
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

#Train model and evaluate
model, best_params = train_model(X_train, y_train)
report, y_pred = evaluate_model(model, X_test, y_test)
print(report)

#Detect anomalies
anomalies = detect_anomalies(X_train, X_test)

#Visualize results
plot_confusion_matrix(y_test, y_pred)
plot_anomalies(X_test, anomalies)

#Save model
save_model(model, 'model.pkl')
save_model(scaler, 'scaler.pkl')
save_model(pca, 'pca.pkl')