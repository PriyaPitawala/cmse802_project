# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def apply_watershed(image: np.ndarray, markers: np.ndarray) -> np.ndarray:
#     """
#     Applies the watershed algorithm to segment regions in the given image.
    
#     Parameters:
#     - image (np.ndarray): Original color image (BGR format).
#     - markers (np.ndarray): Marker image generated from compute_markers().
    
#     Returns:
#     - np.ndarray: Image with watershed segmentation overlayed.
#     """
#     # Convert image to BGR if it is not already
#     if len(image.shape) == 2 or image.shape[2] == 1:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
#     # Make a copy of the original image to overlay results
#     segmented_image = image.copy()
    
#     # Apply watershed algorithm
#     cv2.watershed(segmented_image, markers)
    
#     # Highlight watershed boundaries (where markers == -1)
#     segmented_image[markers == -1] = [255, 0, 0]  # Mark boundaries in red
    
#     return segmented_image

# def display_watershed_result(original_image: np.ndarray, segmented_image: np.ndarray):
#     """
#     Displays the original and segmented images side by side for comparison.
    
#     Parameters:
#     - original_image (np.ndarray): Original color image.
#     - segmented_image (np.ndarray): Image with watershed segmentation overlayed.
#     """
#     fig, ax = plt.subplots(1, 2, figsize=(16,8))
    
#     ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#     ax[0].set_title("Original Image")
#     ax[0].axis("off")
    
#     ax[1].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
#     ax[1].set_title("Watershed Segmented Image")
#     ax[1].axis("off")
    
#     plt.show()


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
    if image is None or markers is None:
        raise ValueError("Input image and markers must be valid numpy arrays.")

    if len(image.shape) < 2 or len(markers.shape) < 2:
        raise ValueError("Invalid image or marker dimensions.")

    # Convert grayscale images to BGR
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Ensure markers are in int32 format as required by OpenCV
    markers = markers.astype(np.int32)

    # Make a copy of the image for visualization
    segmented_image = image.copy()
    
    # Apply watershed algorithm
    cv2.watershed(segmented_image, markers)
    
    # Create an overlay to highlight watershed boundaries
    overlay = np.zeros_like(segmented_image)
    overlay[markers == -1] = [255, 0, 0]  # Mark boundaries in red
    
    # Blend the overlay with the original image
    alpha = 0.6  # Transparency factor
    segmented_image = cv2.addWeighted(segmented_image, 1, overlay, alpha, 0)
    
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
