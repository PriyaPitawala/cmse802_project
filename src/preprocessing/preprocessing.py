import cv2
import numpy as np

class GrayConverter:
    """
    A class to handle grayscale conversion of images.
    """
    @staticmethod
    def convert_to_gray(image: np.ndarray) -> np.ndarray:
        """
        Converts an input image to grayscale.
        
        Parameters:
        - image (np.ndarray): Input image in BGR format.
        
        Returns:
        - np.ndarray: Grayscale image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def equalize_histogram(gray_image: np.ndarray) -> np.ndarray:
        """
        Applies histogram equalization to enhance the contrast of a grayscale image.
        
        Parameters:
        - gray_image (np.ndarray): Grayscale image.
        
        Returns:
        - np.ndarray: Histogram-equalized grayscale image.
        """
        return cv2.equalizeHist(gray_image)
    
def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Applies Gaussian blur to reduce noise in the image.
    
    Parameters:
    - image (np.ndarray): Input image (grayscale or color).
    - kernel_size (int): Size of the Gaussian kernel. Default is 5.
    
    Returns:
    - np.ndarray: Blurred image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Computes the gradient magnitude of an image using Sobel filtering.
    
    Parameters:
    - image (np.ndarray): Input grayscale image.
    
    Returns:
    - np.ndarray: Gradient magnitude image.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    return np.uint8(gradient_magnitude)

def apply_otsu_thresholding(image: np.ndarray) -> np.ndarray:
    """
    Applies Otsu’s thresholding to separate foreground and background.
    
    Parameters:
    - image (np.ndarray): Input grayscale or gradient magnitude image.
    
    Returns:
    - np.ndarray: Binary image after Otsu’s thresholding.
    """
    _, binary_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_thresh

def apply_morphological_opening(image: np.ndarray, kernel_size: int = 3, iterations: int = 2) -> np.ndarray:
    """
    Applies morphological opening to refine segmented regions by removing noise.
    
    Parameters:
    - image (np.ndarray): Binary image after thresholding.
    - kernel_size (int): Size of the structuring element (default is 3x3).
    - iterations (int): Number of times the operation is applied (default is 2).
    
    Returns:
    - np.ndarray: Image after morphological opening.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

def compute_markers(image: np.ndarray) -> np.ndarray:
    """
    Computes foreground and background markers for watershed segmentation.
    
    Parameters:
    - image (np.ndarray): Preprocessed binary image.
    
    Returns:
    - np.ndarray: Marker image with labeled regions.
    """
    # Sure background area (dilation of binary image)
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(image, kernel, iterations=3)
    
    # Distance transform to find sure foreground
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Convert sure foreground to uint8 and find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Create marker labels (background = 0, foreground > 1)
    markers = cv2.connectedComponents(sure_fg)[1]
    markers = markers + 1  # Ensure background is 1, not 0
    markers[unknown == 255] = 0  # Mark unknown regions as 0
    
    return markers

