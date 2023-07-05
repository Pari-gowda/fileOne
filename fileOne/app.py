from flask import Flask
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import io

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('./save_model.h5',compile=False)

# Route for object detection
@app.route('/detect-object/<id>', methods=['POST','GET'])
def detect_pothole(id):
    # Get the image file from the request
    try :
        image_file  =  io.BytesIO(requests.get(f"https://firebasestorage.googleapis.com/v0/b/miniproj-2f595.appspot.com/o/{id}.jpg?alt=media&token=eca9d563-f526-4d9f-b443-72eb653b30d0").content)

        print(f"https://firebasestorage.googleapis.com/v0/b/miniproj-2f595.appspot.com/o/{id}.jpg?alt=media&token=eca9d563-f526-4d9f-b443-72eb653b30d0")

        # Load and preprocess the image
        image = Image.open(image_file)
        image = image.resize((64, 64))
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Debug statements
        print('Image shape:', image.shape)
        print('Image data:', image)

        # Make predictions
        result = model.predict(image)

        # Convert the prediction to a label
        if result[0][0] == 1:
            prediction = 'pothole'
        else:
            prediction = 'Normal'

    except :
        prediction = 'error'

    # Return the prediction as a JSON response
    response = {'prediction': prediction}

    return response

# Run the Flask application
if __name__ == '__main__':
    app.run()