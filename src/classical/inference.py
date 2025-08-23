# SIFT + BFMatcher + Lowe’s ratio + RANSAC → Fundamental Matrix → Submission CSV.
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =====================
# CONFIG
# =====================
DATA_DIR = Path("data")  # adjust if needed
OUTPUT_FILE = "submission.csv"

# =====================
# LOAD SAMPLE SUBMISSION
# =====================
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

# =====================
# HELPER: estimate F with SIFT
# =====================
def estimate_fundamental(img1_path, img2_path):
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return np.zeros((3,3))  # fallback

    # --- Detect & compute descriptors ---
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return np.zeros((3,3))

    # --- Match descriptors ---
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe’s ratio test
    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(pts1) < 8:  # need at least 8 points
        return np.zeros((3,3))

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # --- Estimate fundamental matrix ---
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    if F is None:
        return np.zeros((3,3))

    return F

# =====================
# LOOP OVER TEST SET
# =====================
rows = []
test_csv = pd.read_csv(DATA_DIR / "test.csv")

for _, row in tqdm(test_csv.iterrows(), total=len(test_csv)):
    img1_path = DATA_DIR / "test_images" / row["image_1_id"]
    img2_path = DATA_DIR / "test_images" / row["image_2_id"]

    F = estimate_fundamental(img1_path, img2_path)
    F_flat = " ".join(map(str, F.flatten()))

    rows.append([row["sample_id"], F_flat])

# =====================
# SAVE SUBMISSION
# =====================
df_sub = pd.DataFrame(rows, columns=["sample_id", "fundamental_matrix"])
df_sub.to_csv(OUTPUT_FILE, index=False)
print("Saved submission:", OUTPUT_FILE)

