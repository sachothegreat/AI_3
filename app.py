import os
import cv2
import argparse
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


def load_pretrained_model(model_path, scale):
    """Initialize RealESRGAN with the pretrained model."""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,  # No tiling
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False,  # Use fp16 precision if CUDA is available
    )
    return upsampler


def main():
    parser = argparse.ArgumentParser(description="Image Upscaling using Real-ESRGAN")
    parser.add_argument('--input', type=str, required=True, help="Path to input image or directory")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output images")
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus', help="Model name (e.g., RealESRGAN_x4plus)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pretrained model weights")
    args = parser.parse_args()

    # Check if input is a file or directory
    input_path = args.input
    output_dir = args.output
    model_path = args.model_path
    model_name = args.model_name

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine scale based on model name
    scale = 4 if 'x4' in model_name else 2

    # Load Real-ESRGAN model
    upsampler = load_pretrained_model(model_path, scale)

    # Get list of input images
    if os.path.isdir(input_path):
        input_images = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    else:
        input_images = [input_path]

    # Process each image
    for img_path in input_images:
        print(f"Processing: {img_path}")

        # Load the image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Error: Could not load image {img_path}")
            continue

        try:
            # Enhance the image using Real-ESRGAN
            _, _, upscaled_img = upsampler.enhance(img)
        except RuntimeError as error:
            print(f"Error processing {img_path}: {error}")
            continue

        # Save the upscaled image
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, upscaled_img)
        print(f"Upscaled image saved to {output_path}")


if __name__ == "__main__":
    main()
