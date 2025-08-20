# Image-Matching-Challenge

## **SIFT + OpenCV**:

---

## 🔹 Pipeline (Classical SIFT-based Fundamental Matrix Estimation)

1. **Load the two images** from the dataset.
2. **Detect keypoints & compute descriptors** using SIFT.
3. **Match features** (e.g., BFMatcher + Lowe’s ratio test).
4. **Estimate the fundamental matrix $F$** from correspondences using **RANSAC**.
   * OpenCV: `cv2.findFundamentalMat(matched_points1, matched_points2, cv2.FM_RANSAC)`
5. **Save $F$** in the submission format (flattened row-major).

---

⚠️ Limitations:

* Works well when overlap is good, but may fail under wide baseline, lighting, blur, or repetitive patterns (where deep models help).
* Fundamental matrix can be noisy if correspondences are few.

---

# Reference:
- [Image Matching across Wide Baselines: From Paper to Practice](https://arxiv.org/pdf/2003.01587)


