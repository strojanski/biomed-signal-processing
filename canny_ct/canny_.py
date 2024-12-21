from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import convolve

def resize_image(img, size=(400, 400)):
    """Resize the image to the specified size and save it."""
    img = Image.fromarray(img)
    resized_img = img.resize(size)
    return np.array(resized_img)

def calc_gaussian_kernel(sigma):
    size = int(2 * np.ceil(3 * sigma) + 1)
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def make_positive(angle):
    return angle % 360

def canny_detector(img, improve=False):
    # Load the image
    CT_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    CT_image = resize_image(CT_image, (400, 400))
    
    CT_image = CT_image.astype(np.float64) / 255.0

    # Determine thresholds
    if improve:
        const = .15
        image_mean = np.mean(CT_image)
        image_std = np.std(CT_image) / 2
        lower_bound = max(0, image_mean - image_std)
        upper_bound = min(1, image_mean + image_std)
        threshold_l = max(0, const * lower_bound) 
        threshold_h = min(1, const * upper_bound)
        print(threshold_l, threshold_h)
    else:
        threshold_l = 0.060
        threshold_h = 0.160

    # Apply Gaussian filter
    sigma = min(CT_image.shape) * 0.005
    B = calc_gaussian_kernel(sigma)
    CT_image_gauss = convolve(CT_image, B, mode='reflect')
    
    # Sobel kernels
    # Bx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # By = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Bx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    By = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])



    CT_sobel_x = convolve(CT_image_gauss, Bx, mode='reflect')
    CT_sobel_y = convolve(CT_image_gauss, By, mode='reflect')

    # Edge gradient and angles
    edge_gradient = np.sqrt(CT_sobel_x**2 + CT_sobel_y**2)
    angle = np.arctan2(CT_sobel_y, CT_sobel_x) * (180 / np.pi)
    angle = np.where(angle < 0, angle + 360, angle)

    # Quantize angles to 0, 45, 90, 135
    angle_d = np.zeros_like(angle)
    angle_d[np.logical_or.reduce([
        (angle >= 0) & (angle < 22.5),
        (angle >= 157.5) & (angle < 202.5),
        (angle >= 337.5) & (angle <= 360)
    ])] = 0
    angle_d[np.logical_or.reduce([
        (angle >= 22.5) & (angle < 67.5),
        (angle >= 202.5) & (angle < 247.5)
    ])] = 45
    angle_d[np.logical_or.reduce([
        (angle >= 67.5) & (angle < 112.5),
        (angle >= 247.5) & (angle < 292.5)
    ])] = 90
    angle_d[np.logical_or.reduce([
        (angle >= 112.5) & (angle < 157.5),
        (angle >= 292.5) & (angle < 337.5)
    ])] = 135

    # Non-maximum suppression
    edges = np.zeros_like(edge_gradient)
    for i in range(1, edge_gradient.shape[0] - 1):
        for j in range(1, edge_gradient.shape[1] - 1):
            current = edge_gradient[i, j]
            if angle_d[i, j] == 0:
                neighbors = [edge_gradient[i, j - 1], edge_gradient[i, j + 1]]
            elif angle_d[i, j] == 45:
                neighbors = [edge_gradient[i - 1, j + 1], edge_gradient[i + 1, j - 1]]
            elif angle_d[i, j] == 90:
                neighbors = [edge_gradient[i - 1, j], edge_gradient[i + 1, j]]
            elif angle_d[i, j] == 135:
                neighbors = [edge_gradient[i - 1, j - 1], edge_gradient[i + 1, j + 1]]
            edges[i, j] = current if current >= max(neighbors) else 0

    # Double thresholding and hysteresis
    strong_edges = edges >= threshold_h
    weak_edges = (edges < threshold_h) & (edges >= threshold_l)

    final_edges = np.zeros_like(edges)
    final_edges[strong_edges] = 1

    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                    final_edges[i, j] = 1

    # Save the final result
    final_edges = final_edges.astype(np.uint8)
    return final_edges
