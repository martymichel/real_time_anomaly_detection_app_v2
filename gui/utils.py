"""
Utility functions for image processing and conversions.

Functions:
    - image_to_numpy_rgb: Convert IDS Peak Image or USB ImageView to numpy RGB array
    - numpy_to_qimage: Convert numpy RGB image to QImage
"""

import numpy as np
import cv2
from PySide6.QtGui import QImage
from ids_peak_icv import Image


def image_to_numpy_rgb(img) -> np.ndarray:
    """
    Convert IDS Peak Image or USB ImageView to numpy RGB array.

    Args:
        img: IDS Peak Image object or USBImageView object

    Returns:
        RGB numpy array (H, W, 3) with dtype uint8
    """
    # Check if it's a USB camera frame (has get_numpy_array method)
    if hasattr(img, 'get_numpy_array'):
        # USB camera (OpenCV) - returns BGR
        img_np = img.get_numpy_array()
        # Convert BGR to RGB
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        return img_np

    # Otherwise, it's an IDS Peak Image
    img_np = img.to_numpy_array()

    if img_np.ndim == 1:
        h, w = img.height, img.width
        channels = img.pixel_format.number_of_channels
        if channels == 3:
            img_np = img_np.reshape((h, w, 3))
        else:
            img_np = img_np.reshape((h, w))
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
    elif img_np.ndim == 2:
        img_np = np.stack([img_np, img_np, img_np], axis=-1)

    if img_np.dtype != np.uint8:
        img_np = (img_np / img_np.max() * 255).astype(np.uint8)

    return img_np


def numpy_to_qimage(img_np: np.ndarray) -> QImage:
    """
    Convert numpy RGB image to QImage.

    Args:
        img_np: RGB numpy array (H, W, 3)

    Returns:
        QImage object suitable for Qt display
    """
    h, w, c = img_np.shape

    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    bytes_per_line = c * w
    qimage = QImage(img_np.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    return qimage.copy()  # Make a copy to avoid data corruption
