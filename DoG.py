import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./images/taman.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sigma = 1.6       
k = 1.6            
sigma_k = sigma * k

blur_sigma = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)

blur_sigma_k = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_k)

dog = cv2.subtract(blur_sigma_k, blur_sigma)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(blur_sigma, cmap='gray')
plt.title(r"Gaussian Blur ($\sigma$)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blur_sigma_k, cmap='gray')
plt.title(r"Gaussian Blur ($k\sigma$)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(dog, cmap='gray')
plt.title("Difference of Gaussian (DoG)")
plt.axis('off')

plt.tight_layout()
plt.show()
