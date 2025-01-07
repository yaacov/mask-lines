import numpy as np
from PyQt5.QtGui import QImage


class ImageUtils:
    @staticmethod
    def qimage_to_numpy(qimage):
        """Convert QImage to numpy array safely accounting for bytes per line."""
        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()

        ptr = qimage.bits()
        ptr.setsize(height * bytes_per_line)
        arr = np.frombuffer(ptr, np.uint8)

        # Reshape considering 4 channels (RGBA)
        if bytes_per_line == width * 4:
            return arr.reshape((height, width, 4))

        # If there's padding, we need to remove it
        return arr.reshape(height, bytes_per_line // 4, 4)[:, :width]

    @staticmethod
    def numpy_to_qimage(array):
        """Convert numpy array to QImage ensuring proper alignment."""
        height, width, channels = array.shape
        array = np.require(array, np.uint8, "C")  # Ensure contiguous memory
        qimage = QImage(array.data, width, height, width * 4, QImage.Format_RGB32)
        return qimage.copy()  # Create a deep copy to ensure data ownership
