import numpy as np
from PyQt5.QtGui import QImage, QColor
from skimage.filters import gaussian
from skimage.morphology import thin
from .image_utils import ImageUtils

class ImageProcessor:
    @staticmethod
    def smooth_image(image, pen_color):
        """Apply Gaussian smoothing to the image"""
        if not image:
            return None

        # Convert to numpy array
        image_array = ImageUtils.qimage_to_numpy(image)

        # Create binary mask based on alpha channel (1 where alpha > 0)
        mask = (image_array[:, :, 3] > 0).astype(np.float32)

        # Apply Gaussian smoothing to the mask
        smoothed_mask = gaussian(mask, sigma=1)

        # Create new RGBA array
        height, width = mask.shape
        result = np.zeros((height, width, 4), dtype=np.uint8)

        # Where smoothed mask > 0, set the color and alpha
        mask_threshold = smoothed_mask > 0.1
        result[mask_threshold] = [
            pen_color.red(),
            pen_color.green(),
            pen_color.blue(),
            255,
        ]

        # Convert back to QImage and update
        return QImage(
            result.data, width, height, width * 4, QImage.Format_RGBA8888
        ).copy()

    @staticmethod
    def thin_image(image, pen_color):
        """Apply thinning to the image"""
        if not image:
            return None

        # Convert to numpy array
        image_array = ImageUtils.qimage_to_numpy(image)

        # Create binary mask based on alpha channel
        mask = (image_array[:, :, 3] > 0).astype(np.uint8)

        # Apply thinning
        thinned_mask = thin(mask)

        # Create new RGBA array
        height, width = mask.shape
        result = np.zeros((height, width, 4), dtype=np.uint8)

        # Where thinned mask is True, set the color and alpha
        result[thinned_mask] = [
            pen_color.red(),
            pen_color.green(),
            pen_color.blue(),
            255,
        ]

        # Convert back to QImage
        return QImage(
            result.data, width, height, width * 4, QImage.Format_RGBA8888
        ).copy()

    @staticmethod
    def create_binary_save_image(image):
        """Convert image to binary (black and white) for saving"""
        if not image:
            return None

        # Convert to numpy array
        image_array = ImageUtils.qimage_to_numpy(image)
        
        # Create a new image where alpha > 0 becomes white (255,255,255) and alpha = 0 becomes black (0,0,0)
        bw_array = np.zeros(
            (image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8
        )
        bw_array[image_array[:, :, 3] > 0] = [255, 255, 255]

        # Convert back to QImage
        return QImage(
            bw_array.data,
            bw_array.shape[1],
            bw_array.shape[0],
            bw_array.shape[1] * 3,
            QImage.Format_RGB888,
        )

    @staticmethod
    def smooth_thin_smooth_image(image, pen_color):
        """Apply smooth-thin-smooth filter to the image"""
        if not image:
            return None

        # First smoothing
        image = ImageProcessor.smooth_image(image, pen_color)
        
        # Thinning
        image = ImageProcessor.thin_image(image, pen_color)
        
        # Four more smoothing passes
        for _ in range(4):
            image = ImageProcessor.smooth_image(image, pen_color)
            
        return image
