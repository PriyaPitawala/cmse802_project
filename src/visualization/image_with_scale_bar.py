import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_displayable(image: np.ndarray) -> np.ndarray:
    """
    Converts an image to uint8 format for OpenCV display if needed.
    
    Parameters:
    - image (np.ndarray): The input image.
    
    Returns:
    - np.ndarray: Image in uint8 format suitable for display.
    """
    if image.dtype == np.int32 or image.dtype == np.int64:
        image = image.astype(np.uint8)  # Convert to uint8 for OpenCV compatibility
    elif image.dtype == np.float32 or image.dtype == np.float64:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image

def display_image(image: np.ndarray, title="Image", scale_length_pixels=130, scale_text="60 µm"):
    """
    Displays the image with a scale bar.
    
    Parameters:
    - image (np.ndarray): The image to display.
    - title (str): Title of the displayed image.
    - scale_length_pixels (int): Length of the scale bar in pixels (default: 130 pixels for 60 µm).
    - scale_text (str): Text to display for the scale bar (default: "60 µm").
    """
    # Convert image if necessary
    image = convert_to_displayable(image)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")  # Hide axes
    
    height, width = image.shape[:2]
    
    # Positioning
    x_start = 50  # X-coordinate for scale bar start
    y_start = height - 50  # Y-coordinate (50 px above the bottom)
    
    # Draw single solid white background for both scale bar and text
    background_height = 60  # Adjust height to cover both the bar and text
    rect = plt.Rectangle((x_start - 10, y_start - background_height + 10), 
                         scale_length_pixels + 20, background_height, 
                         color="white", zorder=2)
    ax.add_patch(rect)
    
    # Draw scale bar as a black line
    ax.plot([x_start, x_start + scale_length_pixels], [y_start, y_start], 
            color="black", linewidth=4, zorder=3)
    
    # Adjust text position to align with background
    ax.text(x_start + scale_length_pixels / 2, y_start - background_height / 2 + 5, scale_text,
            color="black", fontsize=12, ha="center", va="center", zorder=4, 
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
    
    plt.show()
