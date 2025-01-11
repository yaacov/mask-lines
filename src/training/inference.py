import argparse
import os
import numpy as np
import torch
from skimage import io
from models.ortho_lines_unet import OrthoLinesUNet


def split_into_patches(img, patch_size=256, step=256):
    """
    Splits an image (H x W x C) into patches, ensuring coverage of edges.
    """
    patches = []
    coords = []
    H, W, C = img.shape

    # Calculate the range for both dimensions to include last row/column
    y_positions = list(range(0, H - patch_size + 1, step))
    x_positions = list(range(0, W - patch_size + 1, step))

    # Add final positions if they're not already included
    if H > patch_size and y_positions[-1] + patch_size < H:
        y_positions.append(H - patch_size)
    if W > patch_size and x_positions[-1] + patch_size < W:
        x_positions.append(W - patch_size)

    for y in y_positions:
        for x in x_positions:
            patch = img[y : y + patch_size, x : x + patch_size, :]
            patches.append(patch)
            coords.append((y, x))

    return patches, coords, H, W


def stitch_patches(patches_pred, coords, H, W, patch_size=256, step=256):
    """
    Rebuild a full image from predicted patches.
    Non-overlapping case => direct placement.
    Overlapping => you'd need to average or blend in overlaps.
    """
    out = np.zeros((H, W, patches_pred[0].shape[2]), dtype=np.float32)
    patch_idx = 0
    for y, x in coords:
        out[y : y + patch_size, x : x + patch_size, :] = patches_pred[patch_idx]
        patch_idx += 1
    return out


def get_device():
    """Get the most suitable device for training."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def main():
    parser = argparse.ArgumentParser(description="Inference using ORTHO line catcher")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image or directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/ortho_lines.pth",
        help="Path to model weights",
    )
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size")
    parser.add_argument("--step", type=int, default=256, help="Step (stride)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = OrthoLinesUNet(
        in_channels=3, out_channels=3, base_features=64, num_convs_per_block=3
    )
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # Create output subdirectories
    for subdir in ["inputs", "targets", "overlay"]:
        os.makedirs(os.path.join(args.output, subdir), exist_ok=True)

    # Check if input is a single file or directory
    if os.path.isdir(args.input):
        fnames = [
            f
            for f in os.listdir(args.input)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for fname in fnames:
            img_path = os.path.join(args.input, fname)
            out_paths = {
                "input": os.path.join(args.output, "inputs", fname),
                "targets": os.path.join(args.output, "targets", fname),
                "overlay": os.path.join(args.output, "overlay", fname),
            }
            infer_and_save(
                img_path, out_paths, model, device, args.patch_size, args.step
            )
    else:
        fname = os.path.basename(args.input)
        out_paths = {
            "input": os.path.join(args.output, "inputs", fname),
            "targets": os.path.join(args.output, "targets", fname),
            "overlay": os.path.join(args.output, "overlay", fname),
        }
        infer_and_save(args.input, out_paths, model, device, args.patch_size, args.step)


def create_overlay(original, prediction, alpha=0.5):
    """
    Create an overlay of the prediction on the original image.
    """
    return original * (1 - alpha) + prediction * alpha


def infer_and_save(img_path, out_paths, model, device, patch_size, step):
    """
    Loads image, splits into patches, runs inference, stitches, saves to multiple directories.
    """
    # Load image and convert to RGB if necessary
    img = io.imread(img_path)

    # Handle different channel configurations
    if img.ndim == 2:  # Grayscale
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:  # RGBA
        img = img[..., :3]  # Keep only RGB channels

    # Save input image
    io.imsave(out_paths["input"], img)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    patches, coords, H, W = split_into_patches(img, patch_size=patch_size, step=step)
    patch_preds = []

    with torch.no_grad():
        for patch in patches:
            patch_tensor = (
                torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(device)
            )
            pred = model(patch_tensor)
            pred = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            patch_preds.append(pred)

    out_img = stitch_patches(
        patch_preds, coords, H, W, patch_size=patch_size, step=step
    )
    out_img = np.clip(out_img, 0, 1)

    # Save targets image
    target_img = (out_img * 255).astype(np.uint8)
    io.imsave(out_paths["targets"], target_img)

    # Create and save overlay
    overlay_img = create_overlay(img, out_img)
    overlay_img = (overlay_img * 255).astype(np.uint8)
    io.imsave(out_paths["overlay"], overlay_img)

    print(f"Saved results to {os.path.dirname(out_paths['input'])}")


if __name__ == "__main__":
    main()
