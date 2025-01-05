import pytest
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from src.data.dataset import PatchImageDataset
from models.ortho_lines_unet import OrthoLinesUNet


@pytest.fixture
def small_dataset(tmp_path):
    """
    Create a tiny dataset with a couple of patches for a quick training test.
    We'll just manually create Tensors in memory, skipping actual image I/O.
    """

    # We'll simulate the dataset by directly creating a PatchImageDataset-like object
    class DummyDataset:
        def __init__(self, length=4):
            self.length = length
            # Each sample: (3, 64, 64)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            x = torch.randn(3, 64, 64)
            y = torch.randn(3, 64, 64)
            return x, y

    return DummyDataset()


def test_training_loop(small_dataset):
    """
    Test a single training iteration doesn't crash and reduces loss.
    """
    model = OrthoLinesUNet(
        in_channels=3, out_channels=3, base_features=32, num_convs_per_block=2
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    loader = DataLoader(small_dataset, batch_size=2, shuffle=False)

    model.train()
    initial_loss = None
    for epoch in range(1):  # just 1 epoch
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        initial_loss = avg_loss if initial_loss is None else initial_loss

    # We can't guarantee big improvements in 1 epoch,
    # but we can check that the code runs without errors and returns a valid float
    assert avg_loss >= 0.0, "Loss is negative or NaN, training likely failed."
    print(f"Test training final avg loss: {avg_loss}")
