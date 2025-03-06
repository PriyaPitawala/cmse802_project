import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def display_image_with_scale(image: np.ndarray, scale_length_pixels: int = 150, scale_text: str = "60 µm"):
    """
    Displays the raw POM image with a scale bar and a single solid white background for both the bar and text.
    
    Parameters:
    - image (np.ndarray): The raw POM image.
    - scale_length_pixels (int): Length of the scale bar in pixels. Default is 150.
    - scale_text (str): Label for the scale bar. Default is "60 µm".
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title("Raw POM Image")
    ax.axis("off")  # Hide axes
    
    # Positioning
    x_start = 50  # X-coordinate for scale bar start
    y_start = image.shape[0] - 50  # Y-coordinate (50 px above the bottom)
    
    # Draw single solid white background for both scale bar and text
    background_height = 60  # Adjust height to cover both the bar and text
    ax.add_patch(plt.Rectangle((x_start - 10, y_start - background_height + 10), scale_length_pixels + 20, background_height, color="white", zorder=2))
    
    # Draw scale bar as a black line
    ax.plot([x_start, x_start + scale_length_pixels], [y_start, y_start], color="black", linewidth=4, zorder=3)
    
    # Add text label with black text on a single solid white background
    ax.text(x_start + scale_length_pixels / 2, y_start - 10, scale_text,
            color="black", fontsize=12, ha="center", va="bottom")
    
    plt.show()