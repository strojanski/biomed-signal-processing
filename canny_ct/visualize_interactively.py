import os
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import cv2

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

def load_images_from_folder(folder, limit=20):
    """Load and preprocess images from a folder."""
    images = []
    for filename in sorted(os.listdir(folder))[:limit]:
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)  # Open image
            img_array = np.array(img)  # Convert to numpy array

            if img_array.ndim == 3:  # If RGB, convert to grayscale
                print(f"Converting image {filename} to grayscale")
                img_array = rgb2gray(img_array)  # Convert to grayscale (float [0, 1])

            print(set(img_array.flatten()))
            # Binarize the grayscale image
            print(f"Processed image {filename}: shape = {img_array.shape}, unique values = {np.unique(img_array)}")
            images.append(img_array)
    return images


def link_edges_24_connectivity(images):
    """Link edges between consecutive 2D images in 3D data using full 24-connectivity."""
    M = len(images)  # Number of slices
    pointcloud = set()  # Use a set to store unique nodes

    def shortest_path_link(x, y, nx, ny, n):
        """Trace the shortest path between two points and set all intermediate pixels."""
        while (x != nx or y != ny):
            if nx > x:
                nx -= 1
            elif nx < x:
                nx += 1
            if ny > y:
                ny -= 1
            elif ny < y:
                ny += 1
            pointcloud.add((nx, ny, n + 1))

    for n in range(M - 1):
        img_n = images[n]
        img_n1 = images[n + 1]

        for y, x in zip(*np.where(img_n == 1)):  # Current layer edge pixels
            # Add the current node to the point cloud
            pointcloud.add((x, y, n))

            # Direct match check
            if img_n1[y, x] == 1:
                pointcloud.add((x, y, n + 1))
                continue

            # Check 3x3 neighborhood
            found = False
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < img_n1.shape[0] and 0 <= nx < img_n1.shape[1] and img_n1[ny, nx] == 1:
                        pointcloud.add((nx, ny, n + 1))
                        found = True
                        break
                if found:
                    break

            # Check 5x5 neighborhood and trace shortest paths
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < img_n1.shape[0] and 0 <= nx < img_n1.shape[1] and img_n1[ny, nx] == 1:
                        shortest_path_link(x, y, nx, ny, n)

    # Convert set to numpy array for visualization
    return np.array(list(pointcloud))




def visualize_pointcloud(pointcloud):
    """Visualize the 3D point cloud interactively with color based on distance from center."""
    print(pointcloud.shape)
    # np.save("pc.npy", pointcloud)

    # Calculate the center of the point cloud
    center = np.mean(pointcloud, axis=0)  # [x_center, y_center, z_center]

    # Calculate distances from the center for coloring
    distances = np.sqrt(np.sum((pointcloud - center) ** 2, axis=1))
    print(pointcloud[:, 2])

    fig = go.Figure(data=[go.Scatter3d(
        x=pointcloud[:, 0],
        y=pointcloud[:, 1],
        z=pointcloud[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=distances, 
            colorscale='cividis',  # Change to deep_r or gray if desired
            opacity=1
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
        title="3D Point Cloud Visualization with Distance-Based Coloring"
    )


    fig.show()

# Example usage
folder_path = 'res/Images-Patient-066259-01/3'  # Change this to your folder with edge images
folder_path = 'body'  # Change this to your folder with edge images

images = load_images_from_folder(folder_path)
pointcloud = link_edges_24_connectivity(images)
visualize_pointcloud(pointcloud)
