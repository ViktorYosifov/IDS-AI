from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

#Loading pre-trained model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

@app.route("/")
def home():
    return "INTRUSION DETECTION SYSTEM"

@app.post("/classify")
def classify():
    data = request.json
    sample = np.array(data['features'])
    sample = scaler.transform(sample)
    sample = pca.transform(sample)
    prediction = model.predict(sample)
    return jsonify({
        'prediction': 'Attack' if prediction[0] == 1 else 'Normal'
    })

if __name__ == '__main__':
    app.run(debug=True)
