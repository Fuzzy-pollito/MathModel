import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import os

# Folder containing images
folder_path = "../Data/toyProblem_F22"

# Get list of PNG files (sorted for consistency)
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

# Load images as grayscale and stack them into a 3D NumPy array
images = np.array([iio.imread(os.path.join(folder_path, f), mode="L") / 255.0 for f in image_files])

frames_to_display = [10, 20, 30]

print(images.shape)  # (num_images, height, width)

if __name__ == "__main__":
    plt.figure()
    for frame in images:
        plt.imshow(frame, cmap="gray")
        plt.axis("off")
        plt.pause(0.01)
        plt.clf()  # Clear figure for next frame
