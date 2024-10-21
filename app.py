import torch
from torchvision.transforms import ToTensor, ToPILImage
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
import io
import traceback
import time  # For measuring inference time

from model import Generator  # Ensure model.py contains the correct Generator architecture

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the Flask app
app = Flask(__name__)

# Allowed file types
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the ESRGAN model
try:
    model = Generator().to(device)
    model.load_state_dict(torch.load('esrgan_final.pth', map_location=device))
    model.eval()
    print("ESRGAN model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    print(f"File received: {file.filename}")

    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        print("File type not allowed")
        return jsonify({
            'error': 'File type not allowed. Only .jpg, .jpeg, and .png files are accepted.'
        }), 400

    try:
        # Read image file
        img = Image.open(file.stream).convert('RGB')
        print(f"Original Image Size: {img.size}")

        # Resize the image
        img = img.resize((256, 256))
        print(f"Resized Image Size: {img.size}")

        # Convert image to tensor
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
        print(f"Image Tensor Shape: {img_tensor.shape}")

        # Run the super-resolution model
        print("Running super-resolution model...")
        start_time = time.time()
        with torch.no_grad():
            high_res_tensor = model(img_tensor)
        end_time = time.time()
        print(f"Inference Time: {end_time - start_time:.2f} seconds")

        # Post-process the result
        high_res_tensor = high_res_tensor.squeeze(0).clamp(0, 1)
        result_img = ToPILImage()(high_res_tensor.cpu())
        print("Image conversion to PIL completed.")

        # Prepare the image for sending
        img_io = io.BytesIO()
        result_img.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        print("Image ready to send")

        # Send the result image to the user
        return send_file(
            img_io,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='result.jpg'  # Use 'download_name' if using Flask >= 2.0
        )

    except Exception as e:
        print(f"Error during image processing: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Image processing failed'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5005)
