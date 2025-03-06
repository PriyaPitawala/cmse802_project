import os
import cv2
import numpy as np
from typing import List, Tuple

def load_pom_image(image_path: str, resize: Tuple[int, int] = None) -> np.ndarray:
    """
    Loads a single POM image without grayscale conversion and resizes if needed.
    
    Parameters:
    - image_path (str): Path to the image file.
    - resize (Tuple[int, int], optional): Resize dimensions (width, height). Default is None.
    
    Returns:
    - np.ndarray: Loaded image array.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")

    # Load image in original color format
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize if specified
    if resize:
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)

    return image

def load_images_from_folder(folder_path: str, resize: Tuple[int, int] = None) -> List[np.ndarray]:
    """
    Loads all images from a specified folder without grayscale conversion.
    
    Parameters:
    - folder_path (str): Directory containing images.
    - resize (Tuple[int, int], optional): Resize dimensions (width, height). Default is None.
    
    Returns:
    - List[np.ndarray]: List of loaded image arrays.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} not found.")

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    images = []

    for file in image_files:
        image_path = os.path.join(folder_path, file)
        try:
            img = load_pom_image(image_path, resize)
            images.append(img)
        except Exception as e:
            print(f"Warning: Skipping {file} due to error: {e}")

    return images

