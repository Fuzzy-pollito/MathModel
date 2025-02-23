import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from Problem1 import images
from Problem1 import frames_to_display


# Function to create a 1D Gaussian kernel
def gaussian_1d(sigma, size=None):
    if size is None:
        size = int(6 * sigma)  # Rule of thumb: 6σ covers >99% of distribution
    if size % 2 == 0:
        size += 1  # Ensure odd size for symmetry

    x = np.arange(-(size // 2), (size // 2) + 1)
    G = np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    return G / G.sum()  # Normalize


# Function to compute the derivative of a 1D Gaussian
def gaussian_derivative_1d(sigma, size=None):
    if size is None:
        size = int(6 * sigma)
    if size % 2 == 0:
        size += 1  # Ensure odd size

    x = np.arange(-(size // 2), (size // 2) + 1)
    G = np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    G_derivative = -x / (sigma**2) * G  # Derivative using chain rule
    return G_derivative - G_derivative.mean()  # Normalize sum to 0


# Set the Gaussian sigma
sigma = 1.5  # Try different values (e.g., 0.5, 1, 2, 3)

# Compute 1D Gaussian and its derivative
G = gaussian_1d(sigma)
G_deriv = gaussian_derivative_1d(sigma)

# Compute gradients using separability
Vx = scipy.ndimage.convolve1d(images, G_deriv, axis=2, mode='nearest')  # X-direction
Vy = scipy.ndimage.convolve1d(images, G_deriv, axis=1, mode='nearest')  # Y-direction
Vt = scipy.ndimage.convolve1d(images, G_deriv, axis=0, mode='nearest')  # Z-direction (time/depth)

# Print the shape to verify
print('Vx shape:', Vx.shape)
print('Vy shape:', Vy.shape)
print('Vt shape:', Vt.shape)

# Display the Gaussian Gradient for selected frames
fig, ax = plt.subplots(len(frames_to_display), 3, figsize=(15, 10))
for i, frame in enumerate(frames_to_display):
    ax[i, 0].imshow(Vx[frame], cmap='gray')
    ax[i, 0].set_title(f"Gaussian Vx (σ={sigma}) - Frame {frame}")
    ax[i, 1].imshow(Vy[frame], cmap='gray')
    ax[i, 1].set_title(f"Gaussian Vy (σ={sigma}) - Frame {frame}")
    if frame < Vt.shape[0]:
        ax[i, 2].imshow(Vt[frame], cmap='gray')
        ax[i, 2].set_title(f"Gaussian Vt (σ={sigma}) - Frame {frame}")
    else:
        ax[i, 2].axis("off")
plt.tight_layout()
plt.show()
