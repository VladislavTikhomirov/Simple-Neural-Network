import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Directory containing the images
image_folder = 'utils/'

# List of image paths
file_dict = ['utils/' + file for file in os.listdir(image_folder) if file.endswith('.png')]

# Number of images to plot
num_images = len(file_dict)

# Create a grid for displaying the images (assuming a 3x3 grid for up to 9 images)
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# Flatten the axes for easy iteration
axes = axes.ravel()

# Plot each image
for i, img_path in enumerate(file_dict):
    img = mpimg.imread(img_path)
    axes[i].imshow(img)
    axes[i].axis('off')  # Hide the axes for a cleaner look

# Hide any unused subplots (if there are fewer than 9 images)
for j in range(i+1, 9):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
