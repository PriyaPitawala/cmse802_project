import cv2
import numpy as np

def split_merged_spherulites(gray_image: np.ndarray,
                              edge_labels: np.ndarray,
                              area_threshold: int = 5000,
                              grad_blur_ksize: int = 3,
                              dist_thresh_factor: float = 0.5) -> np.ndarray:
    """
    Splits large spherulite regions using gradient-based watershed segmentation.

    Parameters:
    - gray_image (np.ndarray): CLAHE-enhanced grayscale image.
    - edge_labels (np.ndarray): Labeled regions from edge-based segmentation.
    - area_threshold (int): Minimum area to consider a region as potentially merged.
    - grad_blur_ksize (int): Kernel size for Sobel gradient.
    - dist_thresh_factor (float): Distance threshold factor for seed region extraction.

    Returns:
    - refined_labels (np.ndarray): Updated label map with merged regions split.
    """
    refined_labels = edge_labels.copy()
    current_max_label = np.max(refined_labels)

    for label_id in np.unique(edge_labels):
        if label_id == 0:
            continue  # skip background

        mask = (edge_labels == label_id).astype(np.uint8)
        region_area = np.sum(mask)

        if region_area < area_threshold:
            continue  # skip small regions

        # Extract region-specific grayscale
        region_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)

        # Gradient magnitude
        grad_x = cv2.Sobel(region_gray, cv2.CV_64F, 1, 0, ksize=grad_blur_ksize)
        grad_y = cv2.Sobel(region_gray, cv2.CV_64F, 0, 1, ksize=grad_blur_ksize)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_mag = cv2.convertScaleAbs(grad_mag)
        grad_mag = cv2.bitwise_and(grad_mag, grad_mag, mask=mask)

        # Distance transform on mask
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, dist_thresh_factor * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Watershed markers
        unknown = cv2.subtract(mask, sure_fg)
        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0

        # Convert grayscale to BGR for watershed
        region_bgr = cv2.cvtColor(region_gray, cv2.COLOR_GRAY2BGR)
        cv2.watershed(region_bgr, markers)

        # Use watershed labels to update refined_labels
        for m_id in np.unique(markers):
            if m_id <= 1:
                continue
            refined_labels[(markers == m_id) & (mask > 0)] = current_max_label + 1
            current_max_label += 1

        # Zero out original merged label to avoid duplication
        refined_labels[edge_labels == label_id] = 0

    return refined_labels
