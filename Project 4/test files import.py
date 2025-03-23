from PIL import Image
import numpy as np
import os

def load_images(folder):
    images = []
    labels = []
    imgforhistogram = []
    for fname in os.listdir(folder):
        img = Image.open(os.path.join(folder, fname)).convert('L')  # grayscale
        img_array = np.array(img).flatten() / 255.0
        #img_array = img_array / 255.0  # normalize and flatten
        if np.mean(img_array) > 0.45:  # likely inverted
            img_array = 1.0 - img_array  # correct it
        else:
            img_array = img_array
        images.append(img_array)
        labels.append(1 if 'positive' in fname.lower() else 0)
    return np.array(images), np.array(labels)

X_train , Y_train = load_images('data - Copy/Train')



print(X_train[6])
print(Y_train[6])

import matplotlib.pyplot as plt

for i in range(20):
    plt.imshow(X_train[i].reshape(224, 224), cmap='gray')
    plt.title(f'Label: {Y_train[0]}')
    plt.show()
