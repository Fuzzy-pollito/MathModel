import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v3 as iio

# Load video frames as grayscale images
folder_path = "../Data/toyProblem_F22"
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
images = np.array([iio.imread(os.path.join(folder_path, f), mode="L") / 255.0 for f in image_files])

# Compute gradients
Vx = images[:, :, 1:] - images[:, :, :-1]  # X gradient
Vy = images[:, 1:, :] - images[:, :-1, :]  # Y gradient
Vt = images[1:, :, :] - images[:-1, :, :]  # Time gradient

# Define parameters
N = 5  # Window size (N x N neighborhood)
frame_idx = 10  # Frame index to analyze
x, y = 50, 100  # Select a pixel position

# Ensure pixel selection is within bounds
half_N = N // 2
if (x - half_N < 0 or x + half_N >= Vx.shape[2] or
    y - half_N < 0 or y + half_N >= Vx.shape[1]):
    raise ValueError("Selected pixel is too close to the boundary!")

# Extract gradients in the neighborhood
Ix = Vx[frame_idx, y - half_N:y + half_N + 1, x - half_N:x + half_N + 1].flatten()
Iy = Vy[frame_idx, y - half_N:y + half_N + 1, x - half_N:x + half_N + 1].flatten()
It = Vt[frame_idx, y - half_N:y + half_N + 1, x - half_N:x + half_N + 1].flatten()

# Construct A and b for least squares
A = np.stack([Ix, Iy], axis=1)  # Shape: (N*N, 2)
b = -It  # Shape: (N*N,)

# Solve for optical flow (u, v)
flow, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
u, v = flow  # Optical flow components

# Plot the selected pixel and flow vector
plt.figure(figsize=(8, 6))
plt.imshow(images[frame_idx], cmap="gray")
plt.quiver(x, y, u, v, color="red", angles="xy", scale_units="xy", scale=1)
plt.scatter(x, y, color="blue", label="Selected Pixel")
plt.legend()
plt.title(f"Optical Flow at Pixel ({x}, {y}) in Frame {frame_idx}")
plt.show()

print(f"Computed flow at ({x}, {y}): u = {u:.4f}, v = {v:.4f}")

