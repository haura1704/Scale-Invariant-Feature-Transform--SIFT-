import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./Scale-Invariant Feature Transform (SIFT)/images/taman.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray, None)

print("Total keypoints:", len(keypoints))


kp = max(keypoints, key=lambda k: k.size)

vis = img_rgb.copy()

x, y = int(kp.pt[0]), int(kp.pt[1])
radius = int(kp.size / 2)

cv2.circle(vis, (x, y), radius, (0, 255, 0), 2)

angle_rad = np.deg2rad(kp.angle)
arrow_length = int(radius * 1.5)

x2 = int(x + arrow_length * np.cos(angle_rad))
y2 = int(y + arrow_length * np.sin(angle_rad))

cv2.arrowedLine(
    vis,
    (x, y),
    (x2, y2),
    (255, 0, 0),
    2,
    tipLength=0.3
)

plt.figure(figsize=(6, 6))
plt.imshow(vis)
plt.title("SIFT Keypoint Orientation (Arrow)")
plt.axis('off')
plt.show()
