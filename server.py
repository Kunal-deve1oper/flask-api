from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Load the pre-trained CNN model
    cnn = tf.keras.models.load_model('model/dog_cat_car_model.keras')

    @app.route('/', methods=['POST'])
    def predict():
        data = request.get_json()
        # print(data)
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        image_data = data['image']
        image_bytes = base64.b64decode(image_data.split(',')[1])  # Remove "data:image/jpeg;base64,"
        image = Image.open(BytesIO(image_bytes))
        test_image = image.resize((64, 64))  # Resize to match model input size
        test_image = np.array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        result = cnn.predict(test_image)
        max_index = np.argmax(result[0])

        if max_index == 0:
            prediction = 'car'
        elif max_index == 1:
            prediction = 'cat'
        else:
            prediction = 'dog'

        return jsonify({'prediction': prediction})

    return app
