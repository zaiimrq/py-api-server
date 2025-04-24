from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model dan label encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Extract all required features
        kategori_umur = float(data["kategori_umur"])
        jenis_kelamin = float(data["jenis_kelamin"])
        protein = float(data["total_protein"])
        lemak = float(data["total_lemak"])
        karbohidrat = float(data["total_karbohidrat"])
        kalori = float(data["total_kalori"])

        # Create input array with all features in correct order
        input_data = np.array(
            [[kategori_umur, jenis_kelamin, protein, lemak, karbohidrat, kalori]]
        )

        # Make prediction
        prediction = model.predict(input_data)
        label = label_encoder.inverse_transform(prediction)

        return jsonify({"status": "success", "klasifikasi": label[0]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
