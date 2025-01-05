import matplotlib.pyplot as plt
import torch


def show_tensor_image(tensor_img, title=None):
    """
    tensor_img: (C, H, W)
    """
    if tensor_img.dim() == 4:  # (N, C, H, W)
        tensor_img = tensor_img[0]
    np_img = tensor_img.permute(1, 2, 0).cpu().numpy()
    plt.imshow(np_img, cmap="gray" if np_img.shape[2] == 1 else None)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
