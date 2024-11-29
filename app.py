import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import GeneratorRRDB  # Ensure this matches the architecture

# Load the pre-trained model for inference
def load_trained_model(model_path):
    model = GeneratorRRDB()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()  # Set model to evaluation mode
    return model

# Preprocess the input low-resolution image
def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.BICUBIC)  # Resize to low-res size if needed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

# Postprocess the output high-resolution image
def postprocess_image(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.detach().cpu().clamp_(0, 1)  # Clamp to [0,1]
    img = transforms.ToPILImage()(tensor)
    return img

# Run the inference
def test_model(model_path, low_res_image_path, output_image_path):
    # Load the model
    generator = load_trained_model(model_path)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)

    # Load and preprocess the low-resolution image
    low_res_img = preprocess_image(low_res_image_path)
    low_res_img = low_res_img.to(device)

    # Perform super-resolution
    with torch.no_grad():
        high_res_img_tensor = generator(low_res_img)

    # Postprocess and save the high-resolution image
    high_res_img = postprocess_image(high_res_img_tensor)
    high_res_img.save(output_image_path)
    print(f"Super-resolved image saved at: {output_image_path}")

    # Display the images
    low_res_img_pil = Image.open(low_res_image_path).convert('RGB')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Low Resolution")
    plt.imshow(low_res_img_pil)

    plt.subplot(1, 2, 2)
    plt.title("Super Resolution")
    plt.imshow(high_res_img)
    plt.show()

if __name__ == "__main__":
    # Paths for the model and images
    model_path = "./weights/RealESRGAN_x4plus.pth"  # Update path to the Real-ESRGAN weights
    low_res_image_path = "./test_images/low_res_image.png"  # Path to low-res image
    output_image_path = "./test_images/high_res_output.png"  # Path to save high-res output

    test_model(model_path, low_res_image_path, output_image_path)
