import argparse
import os
import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


def load_pretrained_model(model_path, scale=4):
    """Load the pretrained Real-ESRGAN model."""
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,  # no tile
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False,  # Use fp16 precision if CUDA is available
    )
    return upsampler


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-ESRGAN Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling scale")
    parser.add_argument("--face_enhance", action="store_true", help="Enable face enhancement")
    return parser.parse_args()


def main():
    """Main function for Real-ESRGAN inference."""
    args = parse_args()
    input_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("Loading pretrained model...")
    upsampler = load_pretrained_model(args.model_path, args.scale)
    print("Model loaded successfully.")

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if not os.path.isfile(img_path):
            continue
        print(f"Processing: {img_path}")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping invalid image: {img_path}")
            continue

        try:
            if args.face_enhance:
                _, _, upscaled_img = upsampler.enhance(img)
            else:
                upscaled_img, _ = upsampler.enhance(img)
        except RuntimeError as e:
            print(f"Error processing {img_path}: {e}")
            print("Try reducing the image size or using a smaller tile size.")
            continue

        output_path = os.path.join(output_dir, f"upscaled_{img_name}")
        cv2.imwrite(output_path, upscaled_img)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
