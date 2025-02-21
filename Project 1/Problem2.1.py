import matplotlib.pyplot as plt
from Problem1 import images
from Problem1 import frames_to_display

# Compute finite difference gradients
Vx = images[:, :, 1:] - images[:, :, :-1]  # Gradient along x (width)
Vy = images[:, 1:, :] - images[:, :-1, :]  # Gradient along y (height)
Vt = images[1:, :, :] - images[:-1, :, :]  # Gradient along t (depth)

# Print shapes to verify
print("Vx shape:", Vx.shape)  # (64, 256, 255)
print("Vy shape:", Vy.shape)  # (64, 255, 256)
print("Vt shape:", Vt.shape)  # (63, 256, 256)

fig, axes = plt.subplots(len(frames_to_display), 3, figsize=(12, 8))

for i, frame in enumerate(frames_to_display):
    axes[i, 0].imshow(Vx[frame], cmap='bwr', aspect='equal')
    axes[i, 0].set_title(f"Vx - Frame {frame}")

    axes[i, 1].imshow(Vy[frame], cmap='bwr', aspect='equal')
    axes[i, 1].set_title(f"Vy - Frame {frame}")

    if frame < Vt.shape[0]:  # Vt has one less frame
        axes[i, 2].imshow(Vt[frame], cmap='bwr', aspect='equal')
        axes[i, 2].set_title(f"Vt - Frame {frame}")
    else:
        axes[i, 2].axis("off")  # Hide if out of range

plt.tight_layout()
plt.show()
