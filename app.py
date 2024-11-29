import argparse
import torch
import os
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Function to load a pre-trained model
def load_pretrained_model(model_path):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False  # Use fp16 precision if CUDA is available
    )
    return upsampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input directory containing low-resolution images")
    parser.add_argument('--output', type=str, required=True, help="Output directory for high-resolution images")
    parser.add_argument('--model_name', type=str, default="RealESRGAN_x4plus", help="Pre-trained model name")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model")
    args = parser.parse_args()

    # Load the upsampler
    upsampler = load_pretrained_model(args.model_path)

    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Process images in the input directory
    for img_name in os.listdir(args.input):
        img_path = os.path.join(args.input, img_name)
        output_path = os.path.join(args.output, f"upscaled_{img_name}")
        if os.path.isfile(img_path):
            try:
                _, _, upscaled_img = upsampler.enhance(img_path)
                upsampler.save_image(upscaled_img, output_path)
                print(f"Processed {img_name} -> {output_path}")
            except RuntimeError as e:
                print(f"Failed to process {img_name}: {str(e)}")

if __name__ == "__main__":
    main()
