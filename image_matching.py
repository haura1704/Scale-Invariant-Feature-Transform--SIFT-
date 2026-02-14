import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('./Scale-Invariant Feature Transform (SIFT)/images/taman.jpg')
img2 = cv2.imread('./Scale-Invariant Feature Transform (SIFT)/images/airmancurtaman.jpg')

if img1 is None or img2 is None:
    print("Error: Gambar tidak ditemukan. Periksa path dataset.")
    exit()

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_matches.append(m)

inliers = 0

if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    matchesMask = mask.ravel().tolist()
    
    inliers = np.sum(matchesMask)
    print(f"Total Good Matches: {len(good_matches)}")
    print(f"Valid Matches (Inliers) setelah RANSAC: {inliers}")
else:
    print("Not enough matches are found - %d/%d" % (len(good_matches), 10))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0), 
                   singlePointColor=None,
                   matchesMask=matchesMask, 
                   flags=2)

matched_img = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good_matches, None, **draw_params)

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.imshow(img1_rgb)
plt.title("Before: Gambar Asli")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img2_rgb)
plt.title("Before: Gambar Potongan")
plt.axis('off')

plt.subplot(2, 2, (3, 4))
plt.imshow(matched_img)
plt.title(f"Final Result: SIFT + RANSAC (Inliers: {inliers})")
plt.axis('off')

plt.tight_layout()
plt.show()

