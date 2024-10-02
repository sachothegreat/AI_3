from flask import Flask, request, jsonify, send_file, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import os

# Load the ESRGAN generator model
model = tf.keras.models.load_model('esrgan_generator.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
print("ESRGAN model loaded successfully")

app = Flask(__name__)

# Allowed file types
ALLOWED_EXTENSIONS = {'mp4', 'jpg', 'jpeg', 'png'}

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve the HTML page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Step 1: Check if the file is being received
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    print(f"File received: {file.filename}")

    # Step 2: Validate file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Only .mp4, .jpg, .jpeg, and .png files are accepted.'}), 400

    # Step 3: Process images (JPG, PNG)
    if file.filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}:
        try:
            img = Image.open(io.BytesIO(file.read()))

            # Step 4: Preprocess the image (Resize to 128x128 and normalize)
            img = np.array(img)
            print(f"Original Image Shape: {img.shape}, Data Type: {img.dtype}, Min-Max: {img.min()}-{img.max()}")

            img_resized = cv2.resize(img, (128, 128))  # Resize image to fit model input
            print(f"Resized Image Shape: {img_resized.shape}, Min-Max: {img_resized.min()}-{img_resized.max()}")

            img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            print(f"Normalized Image Shape: {img_array.shape}, Min-Max: {img_array.min()}-{img_array.max()}")

            # Step 5: Perform super-resolution using the ESRGAN model
            high_res_img = model.predict(img_array)
            print(f"Model Output Shape: {high_res_img.shape}, Min-Max: {high_res_img.min()}-{high_res_img.max()}")

            # Step 6: Post-process the result
            high_res_img = np.squeeze(high_res_img) * 255.0
            high_res_img = np.clip(high_res_img, 0, 255).astype(np.uint8)
            print(f"Post-Processed Image Shape: {high_res_img.shape}, Min-Max: {high_res_img.min()}-{high_res_img.max()}")

            # Step 7: Convert the result to a PIL image and return to the user
            result_img = Image.fromarray(high_res_img)
            img_io = io.BytesIO()
            result_img.save(img_io, 'JPEG')
            img_io.seek(0)
            print("Image ready to send")

            return send_file(img_io, mimetype='image/jpeg')

        except Exception as e:
            print(f"Error during image processing: {str(e)}")
            return jsonify({'error': 'Image processing failed'}), 500

    return jsonify({'error': 'Unsupported file type'}), 400

# Route for serving image from uploads folder
@app.route('/uploads/<filename>')
def get_uploaded_image(filename):
    file_path = os.path.join('uploads', filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg', as_attachment=False)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5005)
