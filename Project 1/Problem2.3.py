from Problem1 import images
#
# sigma = 1
#
# G = lambda x, y, t: 1/(4*np.pi**2*sigma**2)*np.exp(-(x**2+y**2+t**2)/2*sigma**2)
#
# print(G(1, 1, 1))
#


import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

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

# Load your 3D image (64, 256, 256)
# images = np.random.rand(64, 256, 256)  # Uncomment if you need a test array

# Set the Gaussian sigma
sigma = 1.5  # Try different values (e.g., 0.5, 1, 2, 3)

# Compute 1D Gaussian and its derivative
G = gaussian_1d(sigma)
G_deriv = gaussian_derivative_1d(sigma)

# Compute gradients using separability
Vx = scipy.ndimage.convolve1d(images, G_deriv, axis=2, mode='nearest')  # X-direction
Vy = scipy.ndimage.convolve1d(images, G_deriv, axis=1, mode='nearest')  # Y-direction
Vt = scipy.ndimage.convolve1d(images, G_deriv, axis=0, mode='nearest')  # Z-direction (time/depth)

# Visualize results
slice_idx = 32  # Middle slice

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(Vx[slice_idx], cmap='gray')
axes[0].set_title(f"Gaussian Gradient Vx (σ={sigma})")
axes[1].imshow(Vy[slice_idx], cmap='gray')
axes[1].set_title(f"Gaussian Gradient Vy (σ={sigma})")
axes[2].imshow(Vt[slice_idx], cmap='gray')
axes[2].set_title(f"Gaussian Gradient Vt (σ={sigma})")
plt.show()
