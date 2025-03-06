import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_watershed(image: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    Applies the watershed algorithm to segment regions in the given image.
    
    Parameters:
    - image (np.ndarray): Original color image (BGR format).
    - markers (np.ndarray): Marker image generated from compute_markers().
    
    Returns:
    - np.ndarray: Image with watershed segmentation overlayed.
    """
    # Convert image to BGR if it is not already
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Make a copy of the original image to overlay results
    segmented_image = image.copy()
    
    # Apply watershed algorithm
    cv2.watershed(segmented_image, markers)
    
    # Highlight watershed boundaries (where markers == -1)
    segmented_image[markers == -1] = [255, 0, 0]  # Mark boundaries in red
    
    return segmented_image

def display_watershed_result(original_image: np.ndarray, segmented_image: np.ndarray):
    """
    Displays the original and segmented images side by side for comparison.
    
    Parameters:
    - original_image (np.ndarray): Original color image.
    - segmented_image (np.ndarray): Image with watershed segmentation overlayed.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    
    ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    ax[1].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Watershed Segmented Image")
    ax[1].axis("off")
    
    plt.show()
