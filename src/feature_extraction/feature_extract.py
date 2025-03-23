import numpy as np
import pandas as pd
import cv2
from skimage.measure import regionprops, regionprops_table, label
from skimage.segmentation import relabel_sequential


def clean_watershed_labels(markers: np.ndarray) -> np.ndarray:
    """
    Cleans the markers array after watershed by removing -1 boundary pixels
    and relabeling connected components with positive integers.

    Parameters:
    - markers (np.ndarray): Output marker image from cv2.watershed()

    Returns:
    - np.ndarray: Cleaned and relabeled segmented_labels image
    """
    cleaned = markers.copy()
    cleaned[cleaned == -1] = 0
    cleaned[cleaned <= 1] = 0
    cleaned = label(cleaned)
    cleaned, _, _ = relabel_sequential(cleaned)
    return cleaned


def extract_region_features(segmented_labels: np.ndarray, raw_image: np.ndarray) -> pd.DataFrame:
    """
    Extracts morphological and intensity-based features from cleaned watershed labels.

    Parameters:
    - segmented_labels (np.ndarray): Cleaned and labeled image of segmented regions.
    - raw_image (np.ndarray): Original BGR image.

    Returns:
    - pd.DataFrame: DataFrame of features for each region.
    """
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

    props = regionprops_table(
        segmented_labels,
        intensity_image=gray,
        properties=[
            'label',
            'area',
            'equivalent_diameter',
            'perimeter',
            'eccentricity',
            'solidity',
            'orientation',
            'centroid',
            'bbox',
            'mean_intensity',
            'max_intensity',
            'min_intensity',
        ]
    )
    return pd.DataFrame(props)


def overlay_labels_on_image(segmented_labels: np.ndarray, raw_image: np.ndarray) -> np.ndarray:
    """
    Overlays bounding boxes and label text on grayscale-converted image for display.

    Parameters:
    - segmented_labels (np.ndarray): Cleaned label mask.
    - raw_image (np.ndarray): Original BGR image.

    Returns:
    - np.ndarray: RGB image with boxes and label text.
    """
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    props = regionprops(segmented_labels)

    for region in props:
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(overlay, (minc, minr), (maxc, maxr), color=(0, 255, 0), thickness=1)

        centroid_r, centroid_c = region.centroid
        cv2.putText(
            overlay,
            f"{region.label}",
            (int(centroid_c), int(centroid_r)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )

    return overlay
