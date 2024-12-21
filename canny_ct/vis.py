import numpy as np
import plotly.graph_objects as go

def visualize_pointcloud(pointcloud):
    """Visualize the 3D point cloud interactively."""
    
    center = np.mean(pointcloud, axis=0)  # [x_center, y_center, z_center]
    distances = np.sqrt(np.sum((pointcloud - center) ** 2, axis=1))

    fig = go.Figure(data=[go.Scatter3d(
        x=pointcloud[:, 0],
        y=pointcloud[:, 1],
        z=pointcloud[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=distances, 
            colorscale='viridis',  # Change to deep_r or gray if desired
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
    
pc = np.load("pc.npy")
print(pc)
visualize_pointcloud(pc)
