from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_PATH = '/home/alo-vebrisatriadi/Documents/data-engineering/learning/mlops/'

app = Flask(__name__)

def load_model():
    try:
        with open(PROJECT_PATH + 'models/mobile_phone_classification.pkl', 'rb') as file:
            model = pickle.load(file)

        return model
    
    except Exception as e:
        print(f"Error when loading model: {e}")
        return None
    
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)

        return jsonify({
            'status': 'success',
            'prediction': prediction.tolist(),
            'error': None
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'prediction': None,
            'error': str(e)
        })
    
@app.route('/health', methods=['GET'])
def health():
    if model is not None:
        return jsonify({'status': 'healthy', 'message': 'Model is loaded and ready'})
    return jsonify({'status': 'unhealthy', 'message': 'Model not loaded'})

if __name__ == '__main__':
    app.run(debug=True, port=5050)