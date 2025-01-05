import os
import numpy as np
import pytest
import tempfile
from skimage.io import imsave
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.data.dataset import PatchImageDataset


@pytest.fixture
def fake_image_dir(tmpdir):
    """Create fake images for testing"""
    # Create two 256x256 test images with 3 channels (RGB)
    for i in range(2):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img_path = os.path.join(tmpdir, f"test_img_{i}.png")
        Image.fromarray(img).save(img_path)
    return str(tmpdir)


@pytest.fixture
def fake_rgba_image_dir(tmpdir):
    """Create fake RGBA images for testing"""
    # Create two 256x256 test images with 4 channels (RGBA)
    for i in range(2):
        img = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        img_path = os.path.join(tmpdir, f"test_rgba_{i}.png")
        Image.fromarray(img, mode="RGBA").save(img_path)
    return str(tmpdir)


@pytest.fixture
def fake_grayscale_image_dir(tmpdir):
    """Create fake grayscale images for testing"""
    # Create two 256x256 test images with 1 channel
    for i in range(2):
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        img_path = os.path.join(tmpdir, f"test_gray_{i}.png")
        Image.fromarray(img, mode="L").save(img_path)
    return str(tmpdir)


def test_patch_image_dataset(fake_image_dir):
    """
    Test that PatchImageDataset can load images, split into patches, and return correct shapes.
    """
    input_dir = fake_image_dir
    target_dir = fake_image_dir
    patch_size = 128
    step = 128

    dataset = PatchImageDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        patch_size=patch_size,
        step=step,
    )

    # For 256x256 image with patch_size=128 and step=128:
    # - Each image should be split into 4 non-overlapping patches (2x2)
    # - With 2 images, total patches should be 8
    expected_patches = 8  # 2 images * (2 rows * 2 cols)
    assert (
        len(dataset) == expected_patches
    ), f"Expected {expected_patches} patches, got {len(dataset)}"

    # Test patch shape - expect 3 channels (RGB)
    input_patch, target_patch = dataset[0]
    assert input_patch.shape == (
        3,
        patch_size,
        patch_size,
    ), f"Expected shape (3,{patch_size},{patch_size}), got {input_patch.shape}"
    assert target_patch.shape == (3, patch_size, patch_size)

    # Also test DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for batch_x, batch_y in loader:
        assert batch_x.shape == (
            2,
            3,
            patch_size,
            patch_size,
        ), f"Expected batch shape (2,3,{patch_size},{patch_size}), got {batch_x.shape}"
        assert batch_y.shape == (2, 3, patch_size, patch_size)
        break  # just check the first batch


def test_rgba_image_dataset(fake_rgba_image_dir):
    """Test that PatchImageDataset properly handles RGBA images."""
    dataset = PatchImageDataset(
        input_dir=fake_rgba_image_dir,
        target_dir=fake_rgba_image_dir,
        patch_size=128,
        step=128,
    )

    # Test that RGBA images are converted to RGB (3 channels)
    input_patch, target_patch = dataset[0]
    assert input_patch.shape == (
        3,
        128,
        128,
    ), "RGBA should be converted to RGB (3 channels)"
    assert target_patch.shape == (3, 128, 128)

    # Test with DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for batch_x, batch_y in loader:
        assert batch_x.shape == (2, 3, 128, 128), "Batch should have RGB format"
        assert batch_y.shape == (2, 3, 128, 128)
        break


def test_grayscale_image_dataset(fake_grayscale_image_dir):
    """Test that PatchImageDataset properly handles grayscale images."""
    # Test with grayscale preservation
    dataset_gray = PatchImageDataset(
        input_dir=fake_grayscale_image_dir,
        target_dir=fake_grayscale_image_dir,
        patch_size=128,
        step=128,
        keep_grayscale=True,
    )

    # Should be single channel
    input_patch, target_patch = dataset_gray[0]
    assert input_patch.shape == (
        1,
        128,
        128,
    ), "Grayscale should be kept as single channel"
    assert target_patch.shape == (1, 128, 128)

    # Test with RGB conversion (default behavior)
    dataset_rgb = PatchImageDataset(
        input_dir=fake_grayscale_image_dir,
        target_dir=fake_grayscale_image_dir,
        patch_size=128,
        step=128,
        keep_grayscale=False,
    )

    # Should be converted to RGB
    input_patch, target_patch = dataset_rgb[0]
    assert input_patch.shape == (3, 128, 128), "Grayscale should be converted to RGB"
    assert target_patch.shape == (3, 128, 128)
