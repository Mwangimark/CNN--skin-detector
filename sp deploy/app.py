import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('skin_disease_model.h5')

# Define allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file is an allowed image
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function to reshape and normalize
def preprocess_image(img_path):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Resize the image to 28x28 (assuming the model expects this)
    img = cv2.resize(img, (28, 28))

    # Normalize image (if the model was trained on normalized data)
    img = img / 255.0

    # Reshape to (28, 28, 3) and add a batch dimension (1, 28, 28, 3)
    img = np.expand_dims(img, axis=0)
    
    return img

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save file to a temporary location
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Preprocess the image
        img = preprocess_image(file_path)

        # Predict the class
        prediction = model.predict(img)
        class_idx = np.argmax(prediction, axis=1)[0]
        
        # Assuming you have a mapping of classes (you can adjust accordingly)
        classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 
                   'Dermatofibroma', 'Melanocytic nevi', 'Vascular lesions', 'Melanoma']
        
        result = classes[class_idx]
        
        # Return prediction result
        return jsonify({'prediction': result}), 200
    else:
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
