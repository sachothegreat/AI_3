import torch
from PIL import Image
import numpy as np
from model import GeneratorRRDB  # Update import to GeneratorRRDB

# Function to load a single test image
def load_test_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.BICUBIC)
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

# Function to test the generator on a new image
def test_generator(generator_path, image_path, output_path):
    # Initialize GeneratorRRDB model
    generator = GeneratorRRDB()  # Use GeneratorRRDB
    generator.load_state_dict(torch.load(generator_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    generator.eval()  # Set to evaluation mode

    # Load test image
    low_res_image = load_test_image(image_path)
    if torch.cuda.is_available():
        generator = generator.cuda()
        low_res_image = low_res_image.cuda()

    # Generate high-res image
    with torch.no_grad():
        high_res_image = generator(low_res_image)

    # Convert tensor to image and save
    high_res_image = high_res_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0  # Remove batch and channel dim
    high_res_image = np.clip(high_res_image, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(high_res_image)
    result_img.save(output_path)
    print(f"Saved generated high-res image to {output_path}")

if __name__ == "__main__":
    # Test the generator
    test_generator('esrgan_final.pth', 'test_images/low_res_image.png', 'test_images/high_res_output.png')
