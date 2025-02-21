import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from Problem1 import images

# Define Prewitt and Sobel Kernels
prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # Horizontal gradient (Prewitt)
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Vertical gradient (Prewitt)
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Horizontal gradient (Sobel)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Vertical gradient (Sobel)

# # Display the kernels in a single 2x2 grid
# fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# # Prewitt Kernels
# ax[0, 0].imshow(prewitt_x, cmap='gray')
# ax[0, 0].set_title('Prewitt X Kernel')
# ax[0, 1].imshow(prewitt_y, cmap='gray')
# ax[0, 1].set_title('Prewitt Y Kernel')
# # Sobel Kernels
# ax[1, 0].imshow(sobel_x, cmap='gray')
# ax[1, 0].set_title('Sobel X Kernel')
# ax[1, 1].imshow(sobel_y, cmap='gray')
# ax[1, 1].set_title('Sobel Y Kernel')
# # Remove axis ticks for clarity
# for a in ax.ravel():
#     a.axis('off')
# plt.tight_layout()
# plt.show()
print('Prewitt_x: ','\n', prewitt_x)
print('Prewitt_y: ','\n', prewitt_y)
print()
print('Sobel_x: ','\n', sobel_x)
print('Sobel_y: ','\n', sobel_y)

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

# Create a single figure with 2 rows (Prewitt and Sobel) and 3 columns (Vx, Vy, Vt)
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
# Plot Prewitt Results (First Row)
ax[0, 0].imshow(Vx_prewitt[slice_idx], cmap='gray')
ax[0, 0].set_title("Prewitt Vx (Gradient in X)")
ax[0, 1].imshow(Vy_prewitt[slice_idx], cmap='gray')
ax[0, 1].set_title("Prewitt Vy (Gradient in Y)")
ax[0, 2].imshow(Vt_prewitt[slice_idx], cmap='gray')
ax[0, 2].set_title("Prewitt Vt (Gradient in Z)")
# Plot Sobel Results (Second Row)
ax[1, 0].imshow(Vx_sobel[slice_idx], cmap='gray')
ax[1, 0].set_title("Sobel Vx (Gradient in X)")
ax[1, 1].imshow(Vy_sobel[slice_idx], cmap='gray')
ax[1, 1].set_title("Sobel Vy (Gradient in Y)")
ax[1, 2].imshow(Vt_sobel[slice_idx], cmap='gray')
ax[1, 2].set_title("Sobel Vt (Gradient in Z)")
plt.tight_layout()
plt.show()

# Edge detection
# fig, ax = plt.subplots(1, 3, figsize=(15, 10))
# ax[0].imshow(images[slice_idx], cmap='gray')
# ax[0].set_title("Original Image")
# ax[1].imshow(np.sqrt(Vx_prewitt[slice_idx]**2+Vy_prewitt[slice_idx]**2), cmap='gray')
# ax[1].set_title("Prewitt Gradient")
# ax[2].imshow(np.sqrt(Vx_sobel[slice_idx]**2+Vy_sobel[slice_idx]**2), cmap='gray')
# ax[2].set_title("Sobel Gradient")
# for a in ax.ravel():
#     a.axis('off')
# plt.show()
