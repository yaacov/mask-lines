# Using Metrics and Visualization Utilities

## 1. Metrics (e.g., PSNR)

The file `src/utils/metrics.py` contains a function `psnr(pred, target, max_val=1.0)` that calculates the Peak Signal-to-Noise Ratio between predicted and target tensors.

Example usage:
```python
from src.utils.metrics import psnr
import torch

# Suppose you have two Tensors 'pred' and 'target' both of shape (N, C, H, W) or (C, H, W)
score = psnr(pred, target, max_val=1.0)
print("PSNR:", score)
```

## 2. Visualization (show_tensor_image)

In `src/utils/visualization.py`, you'll find `show_tensor_image(tensor_img, title=None)` which uses Matplotlib to display a PyTorch Tensor as an image.

Example usage:
```python
from src.utils.visualization import show_tensor_image
import torch

# Suppose 'output' is a single image Tensor (C, H, W) in range [0,1]
# If 'output' is (N, C, H, W), it will take the first image by default
show_tensor_image(output, title="Model Output")
```

The function:
- Converts the Tensor to a NumPy array
- Calls `matplotlib.pyplot.imshow` to visualize it
- Automatically chooses grayscale if there's only 1 channel
