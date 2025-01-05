import torch
import pytest

from models.ortho_lines_unet import OrthoLinesUNet


def test_deeper_unet_forward():
    """
    Ensures the DeeperUNet forward pass works with a single batch of random data.
    """
    model = OrthoLinesUNet(
        in_channels=3, out_channels=3, base_features=32, num_convs_per_block=3
    )

    # Create a random tensor: batch=2, channels=3, height=256, width=256
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        output = model(x)

    # Check shape => should match (2, 3, 256, 256)
    assert output.shape == (
        2,
        3,
        256,
        256,
    ), f"Output shape is {output.shape}, expected (2, 3, 256, 256)"

    # Just a sanity check that outputs aren't NaN
    assert not torch.isnan(output).any(), "Model output contains NaNs"


def test_deeper_unet_forward_different_input_size():
    """
    Test forward pass with a different size to ensure padding or skip connections work.
    """
    model = OrthoLinesUNet(
        in_channels=3, out_channels=3, base_features=32, num_convs_per_block=3
    )
    # 2,3,130,130 => an odd shape that might require padding in skip connections
    x = torch.randn(2, 3, 130, 130)

    with torch.no_grad():
        output = model(x)

    # Output should match input's spatial size
    assert output.shape == (
        2,
        3,
        130,
        130,
    ), f"Output shape is {output.shape}, expected (2, 3, 130, 130)"
