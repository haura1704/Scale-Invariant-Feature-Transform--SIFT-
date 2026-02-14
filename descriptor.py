import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./images/taman.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

print("Total keypoints:", len(keypoints))

kp = max(keypoints, key=lambda k: k.size)

x, y = int(kp.pt[0]), int(kp.pt[1])
size = int(kp.size)

half = size // 2
patch = gray[
    max(0, y-half):min(gray.shape[0], y+half),
    max(0, x-half):min(gray.shape[1], x+half)
]
patch = cv2.resize(patch, (16, 16))

gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)

magnitude = np.sqrt(gx**2 + gy**2)
orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 360


fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle("SIFT Descriptor: 4×4 Blocks (Orientation Histograms)", fontsize=14)

bin_count = 8

for i in range(4):
    for j in range(4):
        # Ambil blok 4×4
        mag_block = magnitude[i*4:(i+1)*4, j*4:(j+1)*4]
        ori_block = orientation[i*4:(i+1)*4, j*4:(j+1)*4]

        # Histogram orientasi
        hist, _ = np.histogram(
            ori_block,
            bins=bin_count,
            range=(0, 360),
            weights=mag_block
        )

        axes[i, j].bar(range(bin_count), hist)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout()
plt.show()
