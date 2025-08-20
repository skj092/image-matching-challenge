import os
import numpy as np
import cv2
import csv
from glob import glob
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
import random
import sys
from utils import (ReadCovisibilityData, LoadCalibration, ExtractSiftFeatures, ArrayFromCvKps, DrawMatches, ComputeEssentialMatrix, QuaternionFromMatrix, ComputeErrorForOneExample, ComputeMaa)

# A named tuple containing the intrinsics (calibration matrix K) and extrinsics (rotation matrix R, translation vector T) for a given camera.
Gt = namedtuple('Gt', ['K', 'R', 'T'])
eps = 1e-15
os.mkdir('tmp') if not os.path.exists('tmp') else None

src = 'data/train'

val_scenes = []
for f in os.scandir(src):
    if f.is_dir():
        cur_scene = os.path.split(f)[-1]
        val_scenes += [cur_scene]

# Each scene in the validation set contains a list of images, poses, and pairs. Let's pick one and look at some images.
scene = 'colosseum_exterior'
images_dict = {}
for filename in glob(f'{src}/{scene}/images/*.jpg'):
    cur_id = os.path.basename(os.path.splitext(filename)[0])

    # OpenCV expects BGR, but the images are encoded in standard RGB, so you need to do color conversion if you use OpenCV for I/O.
    images_dict[cur_id] = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
print(f'Loaded {len(images_dict)} images.')

# Let's visualize some of the images.
num_rows = 6
num_cols = 4
f, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20), constrained_layout=True)
for i, key in enumerate(images_dict):
    if i >= num_rows * num_cols:
        break
    cur_ax = axes[i % num_rows, i // num_rows]
    cur_ax.imshow(images_dict[key])
    cur_ax.set_title(key)
    cur_ax.axis('off')
plt.savefig(f'tmp/{scene}_images.png')


# Two images from the same scene may not always overlap.
# The dataset contains co-visibility estimates that you can use to find pairs with more or less overlap.
# We recommend using all pairs with a co-visibility estimate of 0.1 or larger.
# For more details, please see Section 3.2 of the paper: https://arxiv.org/abs/2003.01587.

covisibility_dict = ReadCovisibilityData(f'{src}/{scene}/pair_covisibility.csv')

# Let's look at easy pairs first, and difficult pairs later.
easy_subset = [k for k, v in covisibility_dict.items() if v >= 0.7]
difficult_subset = [k for k, v in covisibility_dict.items() if v >= 0.1 and v < 0.2]

for i, subset in enumerate([easy_subset, difficult_subset]):
    print(f'Pairs from an {"easy" if i == 0 else "difficult"} subset')

    for pair in subset[:4]:
        # A pair string is simply two concatenated image IDs, separated with a hyphen.
        image_id_1, image_id_2 = pair.split('-')

        f, axes = plt.subplots(1, 2, figsize=(15, 10), constrained_layout=True)
        axes[0].imshow(images_dict[image_id_1])
        axes[0].set_title(image_id_1)
        axes[1].imshow(images_dict[image_id_2])
        axes[1].set_title(image_id_2)
        for ax in axes:
            ax.axis('off')
        plt.savefig(f'tmp/{scene}_{pair}_easy.png' if i == 0 else f'tmp/{scene}_{pair}_difficult.png')

# Covisibility histogram
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
plt.title('Covisibility histogram')
plt.hist(list(covisibility_dict.values()), bins=10, range=[0, 1])
plt.savefig(f'tmp/{scene}_covisibility_histogram.png')



# The task is finding the relative geometry (rotation, translation) between the two cameras.
# You can read more about epipolar geometry here: https://en.wikipedia.org/wiki/Epipolar_geometry

# This problem is typically (but not always!) solved with sparse features.
# Let's try using SIFT, a seminal work in computer vision (https://en.wikipedia.org/wiki/Scale-invariant_feature_transform).
# No longer the state of the art, but still pretty solid!

num_features = 5000

# You may want to lower the detection threshold, as small images may not be able to reach the budget otherwise.
# Note that you may actually get more than num_features features, as a feature for one point can have multiple orientations (this is rare).
sift_detector = cv2.SIFT_create(num_features, contrastThreshold=-10000, edgeThreshold=-10000)

keys = list(images_dict.keys())
keypoints, descriptors = ExtractSiftFeatures(images_dict[keys[0]], sift_detector, num_features)
print(f'Computed {len(keypoints)} features.')

# Each local feature contains a keypoint (xy, possibly scale, possibly orientation) and a description vector (128-dimensional for SIFT).
image_with_keypoints = cv2.drawKeypoints(images_dict[keys[0]], keypoints, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
fig = plt.figure(figsize=(15, 15))
plt.imshow(image_with_keypoints)
plt.axis('off')
plt.savefig(f'tmp/{scene}_sift_keypoints.png')



# We can find correspondences by brute-force-matching local features between two images. Let's do this for an easy pair.

pair = easy_subset[0]
image_id_1, image_id_2 = pair.split('-')
keypoints_1, descriptors_1 = ExtractSiftFeatures(images_dict[image_id_1], sift_detector, 2000)
keypoints_2, descriptors_2 = ExtractSiftFeatures(images_dict[image_id_2], sift_detector, 2000)

# For each descriptor on one image, find the closest descriptor on the other image.
# With crossCheck=True we keep only bidirectional matches (i.e., two features are nearest neighbours from A to B and also from B to A).
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Compute matches.
cv_matches = bf.match(descriptors_1, descriptors_2)

# Convert keypoints and matches to something more human-readable.
cur_kp_1 = ArrayFromCvKps(keypoints_1)
cur_kp_2 = ArrayFromCvKps(keypoints_2)
matches = np.array([[m.queryIdx, m.trainIdx] for m in cv_matches])

# Plot the brute-force matches.
im_matches = DrawMatches(images_dict[image_id_1], images_dict[image_id_2], cur_kp_1, cur_kp_2, matches)
fig = plt.figure(figsize=(25, 25))
plt.title('Matches before RANSAC')
plt.imshow(im_matches)
plt.axis('off')
plt.savefig(f'tmp/{scene}_{pair}_matches_before_ransac.png')



# Notice that this includes many outliers. We can filter them with a state-of-the-art RANSAC algorithm. References:
# * https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#ga59b0d57f46f8677fb5904294a23d404a
# * https://opencv.org/evaluating-opencvs-new-ransacs

# OpenCV gives us the Fundamental matrix after RANSAC, and a mask over the input matches. The solution is clearly much cleaner, even though it may still contain outliers.
# This F is the prediction you'll submit to the contest.
F, inlier_mask = cv2.findFundamentalMat(cur_kp_1[matches[:, 0]], cur_kp_2[matches[:, 1]], cv2.USAC_MAGSAC, ransacReprojThreshold=0.25, confidence=0.99999, maxIters=10000)

matches_after_ransac = np.array([match for match, is_inlier in zip(matches, inlier_mask) if is_inlier])
im_inliers = DrawMatches(images_dict[image_id_1], images_dict[image_id_2], cur_kp_1, cur_kp_2, matches_after_ransac)
fig = plt.figure(figsize=(25, 25))
plt.title('Matches before RANSAC')
plt.imshow(im_inliers)
plt.axis('off')
plt.savefig(f'tmp/{scene}_{pair}_matches_after_ransac.png')


# Is this any good? Let's load the ground truth.

calib_dict = LoadCalibration(f'{src}/{scene}/calibration.csv')
print(f'Loded ground truth data for {len(calib_dict)} images')
print()

# One important caveat: the scenes were reconstructed from unstructured image collections using Structure-from-Motion (http://colmap.github.io), and are not up to "real-world" scale (i.e. meters, or inches).
# We computed a scaling factor per scene to correct this. This is necessary to compute the metric correctly.

scaling_dict = {}
with open(f'{src}/scaling_factors.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        scaling_dict[row[0]] = float(row[1])

print(f'Scaling factors: {scaling_dict}')
print()

# We can compute the errors now. First, let's decompose the Fundamental matrix we just estimated. TODO explain why we do this.
inlier_kp_1 = ArrayFromCvKps([kp for i, kp in enumerate(keypoints_1) if i in matches_after_ransac[:, 0]])
inlier_kp_2 = ArrayFromCvKps([kp for i, kp in enumerate(keypoints_2) if i in matches_after_ransac[:, 1]])
E, R, T = ComputeEssentialMatrix(F, calib_dict[image_id_1].K, calib_dict[image_id_2].K, inlier_kp_1, inlier_kp_2)
q = QuaternionFromMatrix(R)
T = T.flatten()

# Get the ground truth relative pose difference for this pair of images.
R1_gt, T1_gt = calib_dict[image_id_1].R, calib_dict[image_id_1].T.reshape((3, 1))
R2_gt, T2_gt = calib_dict[image_id_2].R, calib_dict[image_id_2].T.reshape((3, 1))
dR_gt = np.dot(R2_gt, R1_gt.T)
dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()
q_gt = QuaternionFromMatrix(dR_gt)
q_gt = q_gt / (np.linalg.norm(q_gt) + eps)

# Given ground truth and prediction, compute the error for the example above.
err_q, err_t = ComputeErrorForOneExample(q_gt, dT_gt, q, T, scaling_dict[scene])
print(f'Pair "{pair}, rotation_error={err_q:.02f} (deg), translation_error={err_t:.02f} (m)', flush=True)


# # Let's iterate over all the scenes now. Some are much larger than others -- note that the number of pairs increases quadratically with the number of images.
# # We compute the metric for each scene, and then average it over all scenes.
# # For a quick experiment, we cap the number of image pairs for each scene to 50, and show one qualitative example per scene.
#
# show_images = True
# num_show_images = 1
# max_pairs_per_scene = 50
# verbose = True
#
# # We use two different sets of thresholds over rotation and translation. Do not change this -- these are the values used by the scoring back-end.
# thresholds_q = np.linspace(1, 10, 10)
# thresholds_t = np.geomspace(0.2, 5, 10)
#
# # Save the per-sample errors and the accumulated metric to dictionaries, for later inspection.
# errors = {scene: {} for scene in scaling_dict.keys()}
# mAA = {scene: {} for scene in scaling_dict.keys()}
#
# # Instantiate the matcher.
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#
# for scene in scaling_dict.keys():
#     # Load all pairs, find those with a co-visibility over 0.1, and subsample them.
#     covisibility_dict = ReadCovisibilityData(f'{src}/{scene}/pair_covisibility.csv')
#     pairs = [pair for pair, covis in covisibility_dict.items() if covis >= 0.1]
#
#     print(f'-- Processing scene "{scene}": found {len(pairs)} pairs (will keep {min(len(pairs), max_pairs_per_scene)})', flush=True)
#
#     # Subsample the pairs. Note that they are roughly sorted by difficulty (easy ones first), so we shuffle them beforehand: results would be misleading otherwise.
#     random.shuffle(pairs)
#     pairs = pairs[:max_pairs_per_scene]
#
#     # Extract the images in these pairs (we don't need to load images we will not use).
#     ids = []
#     for pair in pairs:
#         cur_ids = pair.split('-')
#         assert cur_ids[0] > cur_ids[1]
#         ids += cur_ids
#     ids = list(set(ids))
#
#     # Load ground truth data.
#     calib_dict = LoadCalibration(f'{src}/{scene}/calibration.csv')
#
#     # Load images and extract SIFT features.
#     images_dict = {}
#     kp_dict = {}
#     desc_dict = {}
#     print('Extracting features...')
#     for id in tqdm(ids):
#         images_dict[id] = cv2.cvtColor(cv2.imread(f'{src}/{scene}/images/{id}.jpg'), cv2.COLOR_BGR2RGB)
#         kp_dict[id], desc_dict[id] = ExtractSiftFeatures(images_dict[id], sift_detector, 2000)
#     print()
#     print(f'Extracted features for {len(kp_dict)} images (avg: {np.mean([len(v) for v in desc_dict.values()])})')
#
#     # Process the pairs.
#     max_err_acc_q_new = []
#     max_err_acc_t_new = []
#     for counter, pair in enumerate(pairs):
#         id1, id2 = pair.split('-')
#
#         # Compute matches by brute force.
#         cv_matches = bf.match(desc_dict[id1], desc_dict[id2])
#         matches = np.array([[m.queryIdx, m.trainIdx] for m in cv_matches])
#         cur_kp_1 = ArrayFromCvKps([kp_dict[id1][m[0]] for m in matches])
#         cur_kp_2 = ArrayFromCvKps([kp_dict[id2][m[1]] for m in matches])
#
#         # Filter matches with RANSAC.
#         F, inlier_mask = cv2.findFundamentalMat(cur_kp_1, cur_kp_2, cv2.USAC_MAGSAC, 0.25, 0.99999, 10000)
#         inlier_mask = inlier_mask.astype(bool).flatten()
#
#         matches_after_ransac = np.array([match for match, is_inlier in zip(matches, inlier_mask) if is_inlier])
#         inlier_kp_1 = ArrayFromCvKps([kp_dict[id1][m[0]] for m in matches_after_ransac])
#         inlier_kp_2 = ArrayFromCvKps([kp_dict[id2][m[1]] for m in matches_after_ransac])
#
#         # Compute the essential matrix.
#         E, R, T = ComputeEssentialMatrix(F, calib_dict[id1].K, calib_dict[id2].K, inlier_kp_1, inlier_kp_2)
#         q = QuaternionFromMatrix(R)
#         T = T.flatten()
#
#         # Get the relative rotation and translation between these two cameras, given their R and T in the global reference frame.
#         R1_gt, T1_gt = calib_dict[id1].R, calib_dict[id1].T.reshape((3, 1))
#         R2_gt, T2_gt = calib_dict[id2].R, calib_dict[id2].T.reshape((3, 1))
#         dR_gt = np.dot(R2_gt, R1_gt.T)
#         dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()
#         q_gt = QuaternionFromMatrix(dR_gt)
#         q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
#
#         # Compute the error for this example.
#         err_q, err_t = ComputeErrorForOneExample(q_gt, dT_gt, q, T, scaling_dict[scene])
#         errors[scene][pair] = [err_q, err_t]
#
#         # Plot the resulting matches and the pose error.
#         if verbose or (show_images and counter < num_show_images):
#             print(f'{pair}, err_q={(err_q):.02f} (deg), err_t={(err_t):.02f} (m)', flush=True)
#         if show_images and counter < num_show_images:
#             im_inliers = DrawMatches(images_dict[id1], images_dict[id2], ArrayFromCvKps(kp_dict[id1]), ArrayFromCvKps(kp_dict[id2]), matches_after_ransac)
#             fig = plt.figure(figsize=(25, 25))
#             plt.title(f'Inliers, "{pair}"')
#             plt.imshow(im_inliers)
#             plt.axis('off')
#             plt.show()
#             print()
#
#     # Histogram the errors over this scene.
#     mAA[scene] = ComputeMaa([v[0] for v in errors[scene].values()], [v[1] for v in errors[scene].values()], thresholds_q, thresholds_t)
#     print()
#     print(f'Mean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')
#     print()
#
# print()
# print('------- SUMMARY -------')
# print()
# for scene in scaling_dict.keys():
#     print(f'-- Mean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')
# print()
# print(f'Mean average Accuracy on dataset: {np.mean([mAA[scene][0] for scene in mAA]):.05f}')
