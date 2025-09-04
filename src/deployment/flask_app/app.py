from flask import Flask, request, jsonify
from src.pipelines.model_service import predict_from_dict

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Real Estate Price Prediction API 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        prediction = predict_from_dict(data)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
