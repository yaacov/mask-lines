import argparse
import yaml
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import PatchImageDataset
from models.ortho_lines_unet import OrthoLinesUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train ORTHO line catcher")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from saved model specified in config",
    )
    return parser.parse_args()


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
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Hyperparams
    epochs = int(config["train"]["epochs"])
    batch_size = int(config["train"]["batch_size"])
    lr = float(config["train"]["learning_rate"])
    patch_size = int(config["train"]["patch_size"])
    step = int(config["train"]["step"])

    # Validate parameters
    if lr <= 0:
        raise ValueError("Learning rate must be positive")
    if epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    # Paths
    train_input_dir = config["paths"]["train_input_dir"]
    train_target_dir = config["paths"]["train_target_dir"]
    model_save_path = config["paths"]["model_save_path"]

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Dataset and Dataloader
    train_dataset = PatchImageDataset(
        input_dir=train_input_dir,
        target_dir=train_target_dir,
        patch_size=patch_size,
        step=step,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Model
    model = OrthoLinesUNet(
        in_channels=3, out_channels=3, base_features=64, num_convs_per_block=3
    )

    # Load existing model if resume flag is set
    if args.resume:
        if not os.path.exists(model_save_path):
            raise ValueError(f"Resume model path not found: {model_save_path}")
        print(f"Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, weights_only=True))

    model = model.to(device)

    # Loss, Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
