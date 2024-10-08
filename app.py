import torch
from torchvision.transforms import ToTensor, ToPILImage
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
import io
import os

# Load the ESRGAN PyTorch generator model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('esrgan_generator.pth', map_location=device)
model.eval()  # Set the model to evaluation mode
print("ESRGAN PyTorch model loaded successfully")

app = Flask(__name__)

# Allowed file types
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

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
        return jsonify({'error': 'File type not allowed. Only .jpg, .jpeg, and .png files are accepted.'}), 400

    # Step 3: Process images (JPG, PNG)
    if file.filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}:
        try:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')

            # Step 4: Preprocess the image (Resize to 128x128 and normalize)
            print(f"Original Image Shape: {img.size}")

            # Convert image to tensor, resize to 128x128
            img_tensor = ToTensor()(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
            print(f"Image Tensor Shape: {img_tensor.shape}")

            # Step 5: Perform super-resolution using the ESRGAN model
            with torch.no_grad():  # Disable gradient calculation for inference
                high_res_tensor = model(img_tensor)

            print(f"Model Output Tensor Shape: {high_res_tensor.shape}")

            # Step 6: Post-process the result (Convert tensor to image)
            high_res_tensor = high_res_tensor.squeeze(0).clamp(0, 1)  # Remove batch dimension and clamp to [0, 1]
            result_img = ToPILImage()(high_res_tensor.cpu())  # Convert back to PIL image

            # Step 7: Send the result image to the user
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
