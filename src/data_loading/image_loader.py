import cv2
import numpy as np

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the given path and returns it as a NumPy array.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - np.ndarray: Loaded image in BGR format.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return image
