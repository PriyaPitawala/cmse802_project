import numpy as np
import matplotlib.pyplot as plt

def display_image(image: np.ndarray, title: str, scale_length_pixels: int = 150, scale_text: str = "60 µm"):
    """
    Displays the preprocessed image with a scale bar and a single solid white background for both the bar and text.
    
    Parameters:
    - image (np.ndarray): The preprocessed image.
    - title (str) : Title of the image
    - scale_length_pixels (int): Length of the scale bar in pixels. Default is 150.
    - scale_text (str): Label for the scale bar. Default is "60 µm".
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
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