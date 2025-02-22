import matplotlib.pyplot as plt
import numpy as np
from Problem1 import images
from Problem2_1 import Vx,Vy,Vt

# Select one pixel in a single frame,p = (x,y,t)
# Extract all gradients in a N×N region around p from Vx,Vy and Vt
import numpy as np

def optical_flow_frame(Vx_frame, Vy_frame, Vt_frame, N=3):  # N is neighborhood size
    height, width = 255,255
    half_N = N // 2
    u = np.zeros((height, width))
    v = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            valid_indices = []
            for i in range(max(0, y - half_N), min(height, y + half_N + 1)):
                for j in range(max(0, x - half_N), min(width, x + half_N + 1)):
                    valid_indices.append((i, j))

            actual_neighborhood_size = len(valid_indices)

            if actual_neighborhood_size > 0:
                A = np.zeros((actual_neighborhood_size, 2))
                b = np.zeros((actual_neighborhood_size, 1))

                for idx, (i, j) in enumerate(valid_indices):
                    A[idx, :] = [Vx_frame[i, j], Vy_frame[i, j]]
                    b[idx, 0] = -Vt_frame[i, j]

                try:
                    u_p = np.linalg.lstsq(A, b, rcond=None)[0]
                    u[y, x] = u_p[0, 0]  # Correct: Assign per-pixel solution
                    v[y, x] = u_p[1, 0]  # Correct: Assign per-pixel solution
                except np.linalg.LinAlgError:
                    u[y, x], v[y, x] = 0, 0
            else:
                u[y, x], v[y, x] = 0, 0

    return u, v

# Example usage (assuming Vx, Vy, Vt are defined):
t = 11  # Example frame index
Vx_frame = Vx[t, :, :]
Vy_frame = Vy[t, :, :]
Vt_frame = Vt[t, :, :]

u, v = optical_flow_frame(Vx_frame, Vy_frame, Vt_frame)

# Visualization (same as before)
flow = np.stack([u, v], axis=-1)
flow_norm = np.linalg.norm(flow, axis=-1)

plt.figure(figsize=(10, 8))
plt.imshow(images[t], cmap='gray')
plt.quiver(np.arange(0, Vx_frame.shape[1], 10), np.arange(0, Vx_frame.shape[0], 10), u[::10, ::10], v[::10, ::10], flow_norm[::10, ::10], cmap='jet', angles='xy', scale_units='xy', scale=0.1, width=0.003)
plt.title(f"Optical Flow at t={t}")
plt.colorbar(label="Magnitude")
plt.show(block=True)