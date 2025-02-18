import numpy as np
import imageio.v3 as iio
%matplotlib qt
import matplotlib.pyplot as plt
import os


folder_path = "/Users/mikkelherskindgudmandsen/PycharmProjects/02526_Mathematical_Modeling/toyProblem_F22"

image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

images = np.array([iio.imread(os.path.join(folder_path, f), mode="L") / 255.0 for f in image_files])

print(images.shape)

plt.ion()  # Turn on interactive mode'

plt.figure()
for frame in images:
    plt.imshow(frame, cmap="gray")  # Display the current frame
    plt.axis("off")  # Hide axes
    plt.pause(0.02)  # Pause for 50ms
    plt.clf()  # Clear figure for next frame

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final figure