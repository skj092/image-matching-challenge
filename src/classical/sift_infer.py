from pathlib import Path
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Load data
data_dir = Path("data/train/taj_mahal")
df = pd.read_csv(os.path.join(data_dir, "pair_covisibility.csv"))

# Pick first pair
pair = df.iloc[0]["pair"]
img1_name, img2_name = pair.split("-")

# Image paths
img1_path = os.path.join(data_dir, "images", img1_name + ".jpg")
img2_path = os.path.join(data_dir, "images", img2_name + ".jpg")

# Load images
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# --- SIFT feature detection ---
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# --- Feature matching with BFMatcher + ratio test (Lowe’s) ---
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.95 * n.distance:
        good_matches.append(m)

print(f"Found {len(good_matches)} good matches")

# --- Draw matches ---
matched_img = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Show result
plt.figure(figsize=(15, 8))
plt.imshow(matched_img, cmap='gray')
plt.axis('off')
plt.savefig("tmp/sift_infer.png")
