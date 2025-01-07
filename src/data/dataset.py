import os
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchImageDataset(Dataset):
    def __init__(
        self,
        input_dir,
        target_dir,
        patch_size=256,
        step=None,
        channels=3,
        keep_grayscale=False,
    ):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.patch_size = patch_size
        self.step = step if step is not None else patch_size
        self.channels = channels
        self.keep_grayscale = keep_grayscale

        # Get list of image files
        self.image_files = sorted(
            [f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        )

        # Pre-calculate patches for each image
        self.patches = []
        for img_file in self.image_files:
            # Load and get image dimensions
            img_path = os.path.join(input_dir, img_file)
            img = Image.open(img_path)
            w, h = img.size

            # Calculate number of patches
            n_patches_w = ((w - self.patch_size) // self.step) + 1
            n_patches_h = ((h - self.patch_size) // self.step) + 1

            logger.debug(
                f"Image {img_file}: size={w}x{h}, patches={n_patches_w}x{n_patches_h}"
            )

            # Store coordinates for each patch
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    x = j * self.step
                    y = i * self.step
                    self.patches.append((img_file, x, y))

    def _ensure_rgb(self, img):
        """Convert image to RGB if needed, respecting keep_grayscale setting"""
        if img.mode == "L":  # If image is grayscale
            return img if self.keep_grayscale else img.convert("RGB")
        elif img.mode == "RGBA":  # If image has alpha channel
            if self.keep_grayscale:
                # Convert to grayscale using alpha as mask
                background = Image.new("L", img.size, 255)
                background.paste(img.convert("L"), mask=img.split()[3])
                return background
            else:
                img_rgb = self._rgba_to_rgb_with_black(img)
                return img_rgb
        elif img.mode != "RGB" and not self.keep_grayscale:
            return img.convert("RGB")
        return img

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_file, x, y = self.patches[idx]

        # Load input image patch
        input_path = os.path.join(self.input_dir, img_file)
        input_img = Image.open(input_path)
        input_img = self._ensure_rgb(input_img)
        input_patch = input_img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # Load target image patch
        target_path = os.path.join(self.target_dir, img_file)
        target_path = self._get_target_path(target_path)
        target_img = Image.open(target_path)
        target_img = self._ensure_rgb(target_img)
        target_patch = target_img.crop((x, y, x + self.patch_size, y + self.patch_size))

        # Convert to tensors
        input_tensor = torch.from_numpy(np.array(input_patch)).float() / 255.0
        target_tensor = torch.from_numpy(np.array(target_patch)).float() / 255.0

        # Move channels to pytorch format (C,H,W)
        if len(input_tensor.shape) == 2:  # Grayscale
            input_tensor = input_tensor.unsqueeze(0)
            target_tensor = target_tensor.unsqueeze(0)
        else:  # RGB
            input_tensor = input_tensor.permute(2, 0, 1)
            target_tensor = target_tensor.permute(2, 0, 1)

        return input_tensor, target_tensor

    def _get_target_path(self, target_path):
        """Find target path, fallback to .png extension if original doesn't exist."""
        import os
        from pathlib import Path

        # Get original path components
        path = Path(target_path)
        target_dir = str(path.parent)
        filename = path.stem

        # Try original path first
        if os.path.exists(target_path):
            return target_path

        # Try .png alternative
        png_path = os.path.join(target_dir, f"{filename}.png")
        if os.path.exists(png_path):
            return png_path

        # Return original if neither exists
        return target_path

    def _rgba_to_rgb_with_black(self, rgba_image):
        """Convert RGBA to RGB with black background where alpha=0"""
        # Split into channels
        r, g, b, a = rgba_image.split()

        # Create black background
        black_background = Image.new("RGB", rgba_image.size, (0, 0, 0))

        # Convert to RGB while preserving alpha blend with black
        rgb_image = Image.composite(rgba_image.convert("RGB"), black_background, a)

        return rgb_image
