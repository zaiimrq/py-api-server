from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan label encoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        protein = data['protein']
        karbo = data['karbohidrat']
        kalori = data['kalori']
        lemak = data['lemak']
        akg = data['akg']

        input_data = np.array([[protein, karbo, kalori, lemak, akg]])
        prediction = model.predict(input_data)
        label = label_encoder.inverse_transform(prediction)

        return jsonify({
            'prediksi': label[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
