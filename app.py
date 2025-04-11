from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from PIL import Image
import numpy as np
import os
import pymongo
import cv2  # OpenCV for face detection
from flask_cors import CORS

app = Flask(__name__, static_folder='static/uploads')

# Enable CORS for frontend
CORS(app, origins="http://127.0.0.1:5500")

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB connection
MONGO_URI = "mongodb+srv://ravikiran6183:Ravi%401717@cluster0.80aqr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client.get_database("pneumonia_db")
collection = db.get_collection("predictions")

# Load the pre-trained chest X-ray detection model
MODEL_PATH = "lung_detection_1.keras"
model = load_model(MODEL_PATH)
print("Model loaded successfully from:", MODEL_PATH)

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Helper function to check for faces
def contains_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

# Helper function to validate chest X-ray
def is_chest_xray(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    
    height, width = image.shape
    aspect_ratio = width / height
    if aspect_ratio < 0.8 or aspect_ratio > 1.2:  # Chest X-rays are often near-square
        return False

    avg_intensity = np.mean(image)
    if avg_intensity < 40 or avg_intensity > 200:  # Chest X-rays have specific intensity ranges
        return False

    return True

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Check for faces
        if contains_face(file_path):
            os.remove(file_path)
            return jsonify({"error": "The uploaded image does not appear to be a valid chest X-ray."
                             "It may contain other body parts or irrelevant content."
                              "Please upload a clear chest X-ray image for analysis."}), 400

        # Check if it is a valid chest X-ray
        if is_chest_xray(file_path):
            # Preprocess the image for prediction
            img = Image.open(file_path).resize((512, 512)).convert('RGB')
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict using the model
            prediction = model.predict(img_array)
            pneumonia_probability = prediction[0][0] * 100
            result_text = (
                f"Pneumonia Detected ({pneumonia_probability:.2f}%)"
                if pneumonia_probability > 50
                else f"No Pneumonia Detected ({100 - pneumonia_probability:.2f}%)"
            )

            # Save the result in MongoDB
            record = {"result": result_text, "filename": file.filename}
            collection.insert_one(record)

            # Return prediction results
            return jsonify({
                "result": result_text,
                "image_url": f"/uploads/{file.filename}"
            })
        else:
            os.remove(file_path)  # Remove invalid file
            return jsonify({"error": "Uploaded image is not a valid chest X-ray."}), 400

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "An error occurred during prediction."}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
