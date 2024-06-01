import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(url, column_names):
    df = pd.read_csv(url, names=column_names)
    return df

def encode_features(df, categorical_cols):
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def preprocessing_data(df):
    X = df.drop(columns=['label'])
    y = df['label'].apply(lambda x: 1 if x != 'normal.' else 0)  #Binary classification, 1 - attack, 0 - normal
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler
