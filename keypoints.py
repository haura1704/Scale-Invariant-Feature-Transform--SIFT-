import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./images/taman.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray, None)

print("Jumlah keypoints:", len(keypoints))

img_kp = cv2.drawKeypoints(
    img_rgb,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10, 6))
plt.imshow(img_kp)
plt.title("SIFT Keypoints (Scale & Orientation)")
plt.axis('off')
plt.show()
