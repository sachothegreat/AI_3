import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def load_pretrained_model(model_path, scale=4):
    """
    Load the Real-ESRGAN pretrained model.
    """
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    return RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,  # Disable tile processing for large images
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False,  # Use fp16 precision if CUDA is available
    )

def upscale_image(input_image_path, output_image_path, upsampler):
    """
    Upscale a single image using Real-ESRGAN.
    """
    img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    try:
        # Upscale the image
        output, _ = upsampler.enhance(img, outscale=4)
        cv2.imwrite(output_image_path, output)
        print(f"Upscaled image saved to: {output_image_path}")
    except RuntimeError as e:
        print(f"Error during upscaling: {e}")

if __name__ == "__main__":
    # Define paths
    model_path = "/content/AI_3/weights/RealESRGAN_x4plus.pth"  # Pre-trained model path
    input_image_path = "/content/AI_3/low_res/600.png"  # Path to the low-res image
    output_image_path = "/content/AI_3/high_res_output.png"  # Path to save the high-res output

    # Load the Real-ESRGAN model
    upsampler = load_pretrained_model(model_path)

    # Perform image upscaling
    upscale_image(input_image_path, output_image_path, upsampler)
