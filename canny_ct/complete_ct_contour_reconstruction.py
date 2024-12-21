from canny_ import canny_detector, resize_image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

original_imgs = []
detected_edges = []
connectivity_edges = []

def process_img(img):
    img = resize_image(img, (400, 400))
    img = canny_detector(img)
    return img

def process_stack(path):
    for i, img in enumerate(os.listdir(path)):
        img_path = path + img
        
        img = cv2.imread(img_path)
        original_imgs.append(img)
        
        img = process_img(img)
        detected_edges.append(img)
        os.makedirs(f"processed/", exist_ok=True)
        cv2.imwrite(f"processed/{i:04d}.png", img)
        

def link_edges_3d(edges_3d):
    """Link edges between consecutive 2D slices in 3D using 24-connectivity."""
    linked_edges_3d = np.copy(edges_3d)
    
    for z in range(1, edges_3d.shape[0] - 1):  # Exclude first and last slices
        for y in range(1, edges_3d.shape[1] - 1):
            for x in range(1, edges_3d.shape[2] - 1):
                if edges_3d[z, y, x] == 1:
                    for dz in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dz == 0 and dy == 0 and dx == 0:
                                    continue
                                if (0 <= z + dz < edges_3d.shape[0] and
                                    0 <= y + dy < edges_3d.shape[1] and
                                    0 <= x + dx < edges_3d.shape[2] and
                                    edges_3d[z + dz, y + dy, x + dx] == 1):
                                    linked_edges_3d[z, y, x] = 1
                                    break
    return linked_edges_3d

def visualize_edge_sequences(original_images, linked_images, final_linked_images, limit=10):
    """Visualize the sequence of original, linked, and final linked images."""
    num_images = len(original_images)
    
    # Create subplots to show the images
    fig, axes = plt.subplots(limit, 3, figsize=(12, num_images * 4))
    
    for i in range(limit):
        # Original image with edges
        axes[i, 0].imshow(original_images[i], cmap='gray')
        axes[i, 0].set_title(f"Original Edges - Slice {i+1}", color="red")
        axes[i, 0].axis('off')
        
        # Image after edge linking within the slice
        axes[i, 1].imshow(linked_images[i], cmap='gray')
        axes[i, 1].set_title(f"Linked Edges - Slice {i+1}", color="red")
        axes[i, 1].axis('off')
        
        # Final image after 24-connectivity edge linking
        axes[i, 2].imshow(final_linked_images[i], cmap='gray')
        axes[i, 2].set_title(f"Final Linked Edges - Slice {i+1}", color="red")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
        
def show_imgs(imgs, title, limit=10):
    if type(imgs) == list:
        
        fig, ax = plt.subplots(limit//2, 2, figsize=(10, 10))
        
        for i in range(limit//2):
            for j in range(2):
                ax[i][j].imshow(imgs[i+j], cmap='gray')
                
        plt.title(title)
            
        plt.show()
            
    else:
       plt.imshow(imgs, cmap='gray')
       plt.title(title)
       plt.show()
    
if __name__ == "__main__":
    process_stack("data/download/Images-Patient-000302-01/1.3.12.2.1107.5.1.4.50454.30000008011507012606200000061/2/")

    pointcloud = link_edges_3d(np.array(detected_edges))
    connectivity_edges = pointcloud
    
    print(pointcloud)
    
    # show_imgs(original_imgs, title="Original", limit=10)
    # show_imgs(detected_edges, title="Detected Edges", limit=10)
    visualize_edge_sequences(np.array(original_imgs), np.array(detected_edges), connectivity_edges)
    
    
