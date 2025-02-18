import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from Problem1 import images

# Define Prewitt and Sobel Kernels
prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # Horizontal gradient (Prewitt)
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Vertical gradient (Prewitt)
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Horizontal gradient (Sobel)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Vertical gradient (Sobel)

# Compute gradients using Prewitt filter
Vx_prewitt = scipy.ndimage.convolve(images, prewitt_x[np.newaxis, :, :])  # X-gradient (Prewitt)
Vy_prewitt = scipy.ndimage.convolve(images, prewitt_y[np.newaxis, :, :])  # Y-gradient (Prewitt)
Vt_prewitt = scipy.ndimage.prewitt(images, axis=0)  # Z-gradient (Prewitt)

# Compute gradients using Sobel filter
Vx_sobel = scipy.ndimage.convolve(images, sobel_x[np.newaxis, :, :])  # X-gradient (Sobel)
Vy_sobel = scipy.ndimage.convolve(images, sobel_y[np.newaxis, :, :])  # Y-gradient (Sobel)
Vt_sobel = scipy.ndimage.sobel(images, axis=0)  # Z-gradient (Sobel)

# Choose a middle slice for visualization
slice_idx = 32  # Middle slice

# Plot Prewitt Results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(Vx_prewitt[slice_idx], cmap='gray')
axes[0].set_title("Prewitt Vx (Gradient in X)")
axes[1].imshow(Vy_prewitt[slice_idx], cmap='gray')
axes[1].set_title("Prewitt Vy (Gradient in Y)")
axes[2].imshow(Vt_prewitt[slice_idx], cmap='gray')
axes[2].set_title("Prewitt Vt (Gradient in Z)")
plt.show()

# Plot Sobel Results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(Vx_sobel[slice_idx], cmap='gray')
axes[0].set_title("Sobel Vx (Gradient in X)")
axes[1].imshow(Vy_sobel[slice_idx], cmap='gray')
axes[1].set_title("Sobel Vy (Gradient in Y)")
axes[2].imshow(Vt_sobel[slice_idx], cmap='gray')
axes[2].set_title("Sobel Vt (Gradient in Z)")
plt.show()
