from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model
with open("model/classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Load the label encoder
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load split information
with open("model/split_info.pkl", "rb") as f:
    split_info = pickle.load(f)

app = Flask(__name__)

# API: Get Model Training Status
@app.route("/get-status", methods=["GET"])
def get_status():
    return jsonify(split_info)

# API: Predict class for input features
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])  # Convert input to DataFrame

        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
