import os
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import cv2

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

def load_images_from_folder(folder):
    """Load and preprocess images from a folder."""
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)  # Open image
            img_array = np.array(img)  # Convert to numpy array

            if img_array.ndim == 3:  # If RGB, convert to grayscale
                print(f"Converting image {filename} to grayscale")
                img_array = rgb2gray(img_array)  # Convert to grayscale (float [0, 1])

            # Binarize the grayscale image
            thresh = threshold_otsu(img_array)  # Compute Otsu threshold
            binary_img = (img_array > thresh).astype(int)  # Binarize image
            print(f"Processed image {filename}: shape = {binary_img.shape}, unique values = {np.unique(binary_img)}")
            images.append(binary_img)
    return images


def link_edges_24_connectivity(images):
    """Link edges between consecutive 2D images in 3D data using 24-connectivity."""
    M = len(images)
    pointcloud = set()  # Use a set to store unique nodes

    for n in range(M - 1):
        img_n = images[n]
        img_n1 = images[n + 1]

        for y, x in zip(*np.where(img_n == 1)):  # Current layer edge pixels
            # Add the current node to the point cloud
            pointcloud.add((x, y, n))

            # Check direct match in the next layer
            if img_n1[y, x] == 1:
                pointcloud.add((x, y, n + 1))
                continue

            # Check 3x3 neighborhood in the next layer
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < img_n1.shape[0] and 0 <= nx < img_n1.shape[1]:
                        if img_n1[ny, nx] == 1:
                            pointcloud.add((nx, ny, n + 1))
                            break

            # Check 5x5 neighborhood in the next layer
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < img_n1.shape[0] and 0 <= nx < img_n1.shape[1]:
                        if img_n1[ny, nx] == 1:
                            pointcloud.add((nx, ny, n + 1))

    # Convert set to numpy array for visualization
    return np.array(list(pointcloud))



def visualize_pointcloud(pointcloud):
    """Visualize the 3D point cloud interactively."""
    print(pointcloud.shape)
    np.save("pc.npy", pointcloud)
    fig = go.Figure(data=[go.Scatter3d(
        x=pointcloud[:, 0],
        y=pointcloud[:, 1],
        z=pointcloud[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=pointcloud[:, 2],  # Use z-axis for coloring
            colorscale='Cividis',
            opacity=0.8
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(backgroundcolor="white"),
            yaxis=dict(backgroundcolor="white"),
            zaxis=dict(backgroundcolor="white"),
        ),
        title="3D Point Cloud Visualization with 24-Connectivity"
    )

    fig.show()

# Example usage
folder_path = 'res/Images-Patient-000302-01/2'  # Change this to your folder with edge images
images = load_images_from_folder(folder_path)
pointcloud = link_edges_24_connectivity(images)
visualize_pointcloud(pointcloud)
