import os
import numpy as np
from PIL import Image
import plotly.graph_objects as go

def load_images_from_folder(folder):
    """Load all edge images from a folder."""
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            images.append(np.array(img))
    return images

def create_pointcloud_from_images(images):
    """Convert stacked edge images into a 3D point cloud."""
    pointcloud = []
    
    for z, img in enumerate(images):
        # Find coordinates of edge pixels
        y, x = np.where(img > 0)  # Non-zero pixels (edges)
        z_values = np.full_like(x, z)  # Z-coordinate is the image index
        
        # Combine the coordinates into 3D points
        points = np.column_stack((x, y, z_values))
        pointcloud.append(points)
    
    # Stack all 2D point clouds to form the final 3D point cloud
    pointcloud = np.vstack(pointcloud)
    return pointcloud

def visualize_pointcloud(pointcloud):
    """Visualize the 3D point cloud interactively."""
    
    center = np.mean(pointcloud, axis=0)  # [x_center, y_center, z_center]
    distances = np.sqrt(np.sum((pointcloud - center) ** 2, axis=1))

    print(pointcloud.shape)
    pointcloud[:, 2] = -pointcloud[:, 2]

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
            zaxis_title='Z'
        ),
        title="Interactive 3D Point Cloud from Edge Images"
    )

    fig.show()

# Example usage
folder_path = 'res/Images-Patient-000302-01/2'  # Change this to your folder with edge images
images = load_images_from_folder(folder_path)
pointcloud = create_pointcloud_from_images(images)
visualize_pointcloud(pointcloud)
