from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions


app = Flask(__name__)
CORS(app)

model = MobileNetV3Small(weights="imagenet")


@app.route("/", methods=["POST"])
def predict():
    data = request.get_json()
    # print(data)
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    image_data = data["image"]
    image_bytes = base64.b64decode(
        image_data.split(",")[1]
    )  # Remove "data:image/jpeg;base64,"
    image = Image.open(BytesIO(image_bytes))
    test_image = image.resize((224, 224))  # Resize to match model input size
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    print("hello")
    x = preprocess_input(test_image)

    # Make predictions
    predictions = model.predict(x)

    # Decode predictions
    decoded_predictions = decode_predictions(predictions, top=5)[0]

    # Format predictions as JSON
    results = [
        {"label": label, "score": float(score)}
        for (_, label, score) in decoded_predictions
    ]

    return jsonify({"predictions": results})

@app.route("/", methods=["GET"])
def test():
    return jsonify({"data" : "hELLO"})


if __name__ == '__main__':
    app.run(debug=False)