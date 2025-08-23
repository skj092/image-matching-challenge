import numpy as np
import cv2
import csv
from collections import namedtuple
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

Gt = namedtuple('Gt', ['K', 'R', 'T'])
eps = 1e-15


def ReadCovisibilityData(filename):
    covisibility_dict = {}
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            covisibility_dict[row[0]] = float(row[1])

    return covisibility_dict


def NormalizeKeypoints(keypoints, K):
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints


def ComputeEssentialMatrix(F, K1, K2, kp1, kp2):
    '''Compute the Essential matrix from the Fundamental matrix, given the calibration matrices. Note that we ask participants to estimate F, i.e., without relying on known intrinsics.'''

    # Warning! Old versions of OpenCV's RANSAC could return multiple F matrices, encoded as a single matrix size 6x3 or 9x3, rather than 3x3.
    # We do not account for this here, as the modern RANSACs do not do this:
    # https://opencv.org/evaluating-opencvs-new-ransacs
    assert F.shape[0] == 3, 'Malformed F?'

    # Use OpenCV's recoverPose to solve the cheirality check:
    # https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0
    E = np.matmul(np.matmul(K2.T, F), K1).astype(np.float64)

    kp1n = NormalizeKeypoints(kp1, K1)
    kp2n = NormalizeKeypoints(kp2, K2)
    num_inliers, R, T, mask = cv2.recoverPose(E, kp1n, kp2n)

    return E, R, T


def ArrayFromCvKps(kps):
    '''Convenience function to convert OpenCV keypoints into a simple numpy array.'''

    return np.array([kp.pt for kp in kps])


def QuaternionFromMatrix(matrix):
    '''Transform a rotation matrix into a quaternion.'''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
              [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
              [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
              [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0

    # The quaternion is the eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0:
        np.negative(q, q)

    return q


def ExtractSiftFeatures(image, detector, num_features):
    '''Compute SIFT features for a given image.'''

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kp, desc = detector.detectAndCompute(gray, None)
    return kp[:num_features], desc[:num_features]


def ComputeErrorForOneExample(q_gt, T_gt, q, T, scale):
    '''Compute the error metric for a single example.

    The function returns two errors, over rotation and translation. These are combined at different thresholds by ComputeMaa in order to compute the mean Average Accuracy.'''

    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return err_q * 180 / np.pi, err_t


def ComputeMaa(err_q, err_t, thresholds_q, thresholds_t):
    '''Compute the mean Average Accuracy at different tresholds, for one scene.'''

    assert len(err_q) == len(err_t)

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)


def BuildCompositeImage(im1, im2, axis=1, margin=0, background=1):
    '''Convenience function to stack two images with different sizes.'''

    if background != 0 and background != 1:
        background = 1
    if axis != 0 and axis != 1:
        raise RuntimeError('Axis must be 0 (vertical) or 1 (horizontal')

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape

    if axis == 1:
        composite = np.zeros((max(h1, h2), w1 + w2 + margin, 3), dtype=np.uint8) + 255 * background
        if h1 > h2:
            voff1, voff2 = 0, (h1 - h2) // 2
        else:
            voff1, voff2 = (h2 - h1) // 2, 0
        hoff1, hoff2 = 0, w1 + margin
    else:
        composite = np.zeros((h1 + h2 + margin, max(w1, w2), 3), dtype=np.uint8) + 255 * background
        if w1 > w2:
            hoff1, hoff2 = 0, (w1 - w2) // 2
        else:
            hoff1, hoff2 = (w2 - w1) // 2, 0
        voff1, voff2 = 0, h1 + margin
    composite[voff1:voff1 + h1, hoff1:hoff1 + w1, :] = im1
    composite[voff2:voff2 + h2, hoff2:hoff2 + w2, :] = im2

    return (composite, (voff1, voff2), (hoff1, hoff2))


def DrawMatches(im1, im2, kp1, kp2, matches, axis=1, margin=0, background=0, linewidth=2):
    '''Draw keypoints and matches.'''

    composite, v_offset, h_offset = BuildCompositeImage(im1, im2, axis, margin, background)

    # Draw all keypoints.
    for coord_a, coord_b in zip(kp1, kp2):
        composite = cv2.drawMarker(composite, (int(coord_a[0] + h_offset[0]), int(coord_a[1] + v_offset[0])), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)
        composite = cv2.drawMarker(composite, (int(coord_b[0] + h_offset[1]), int(coord_b[1] + v_offset[1])), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)

    # Draw matches, and highlight keypoints used in matches.
    for idx_a, idx_b in matches:
        composite = cv2.drawMarker(composite, (int(kp1[idx_a, 0] + h_offset[0]), int(kp1[idx_a, 1] + v_offset[0])), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        composite = cv2.drawMarker(composite, (int(kp2[idx_b, 0] + h_offset[1]), int(kp2[idx_b, 1] + v_offset[1])), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        composite = cv2.line(composite,
                             tuple([int(kp1[idx_a][0] + h_offset[0]),
                                   int(kp1[idx_a][1] + v_offset[0])]),
                             tuple([int(kp2[idx_b][0] + h_offset[1]),
                                   int(kp2[idx_b][1] + v_offset[1])]), color=(0, 0, 255), thickness=1)
    return composite


def LoadCalibration(filename):
    '''Load calibration data (ground truth) from the csv file.'''

    calib_dict = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue

            camera_id = row[0]
            K = np.array([float(v) for v in row[1].split(' ')]).reshape([3, 3])
            R = np.array([float(v) for v in row[2].split(' ')]).reshape([3, 3])
            T = np.array([float(v) for v in row[3].split(' ')])
            calib_dict[camera_id] = Gt(K=K, R=R, T=T)

    return calib_dict


def load_images(scene_path):
    images = {}
    for filename in glob(f"{scene_path}/images/*.jpg"):
        img_id = os.path.splitext(os.path.basename(filename))[0]
        images[img_id] = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    return images

def match_and_filter(kp1, desc1, kp2, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = np.array([[m.queryIdx, m.trainIdx] for m in bf.match(desc1, desc2)])

    pts1 = ArrayFromCvKps([kp1[m[0]] for m in matches])
    pts2 = ArrayFromCvKps([kp2[m[1]] for m in matches])

    F, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.USAC_MAGSAC, 0.25, 0.99999, 10000
    )
    if F is None:
        return None, None, None, None

    mask = mask.astype(bool).flatten()
    inlier_matches = matches[mask]
    inlier_pts1 = ArrayFromCvKps([kp1[m[0]] for m in inlier_matches])
    inlier_pts2 = ArrayFromCvKps([kp2[m[1]] for m in inlier_matches])
    return F, inlier_pts1, inlier_pts2, inlier_matches

def compute_pose_error(F, kp1, kp2, calib, id1, id2, scale):
    E, R, T = ComputeEssentialMatrix(F, calib[id1].K, calib[id2].K, kp1, kp2)
    q_pred = QuaternionFromMatrix(R)
    T_pred = T.flatten()

    R1, T1 = calib[id1].R, calib[id1].T.reshape((3, 1))
    R2, T2 = calib[id2].R, calib[id2].T.reshape((3, 1))
    dR_gt = R2 @ R1.T
    dT_gt = (T2 - dR_gt @ T1).flatten()
    q_gt = QuaternionFromMatrix(dR_gt)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)

    return ComputeErrorForOneExample(q_gt, dT_gt, q_pred, T_pred, scale)

def load_scaling_factors(path):
    scaling = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            scaling[row[0]] = float(row[1])
    return scaling

# ----------------------------------------------------
# Scene processing
# ----------------------------------------------------
def process_scene(scene, src, sift, scaling_dict, show_matches=False):
    scene_path = f"{src}/{scene}"
    covisibility = ReadCovisibilityData(f"{scene_path}/pair_covisibility.csv")
    calib = LoadCalibration(f"{scene_path}/calibration.csv")

    pairs = [p for p, cov in covisibility.items() if cov >= 0.1]
    random.shuffle(pairs)
    pairs = pairs[:50]  # cap

    ids = list(set(sum([p.split("-") for p in pairs], [])))
    images = {i: cv2.cvtColor(cv2.imread(f"{scene_path}/images/{i}.jpg"), cv2.COLOR_BGR2RGB) for i in ids}
    kp, desc = {}, {}
    for i in tqdm(ids, desc=f"Extracting features [{scene}]"):
        kp[i], desc[i] = ExtractSiftFeatures(images[i], sift, 2000)

    errors = {}
    for pair in pairs:
        id1, id2 = pair.split("-")
        F, in1, in2, inlier_matches = match_and_filter(kp[id1], desc[id1], kp[id2], desc[id2])
        if F is None:
            continue

        err_q, err_t = compute_pose_error(F, in1, in2, calib, id1, id2, scaling_dict[scene])
        errors[pair] = (err_q, err_t)

        if show_matches:
            im = DrawMatches(images[id1], images[id2], ArrayFromCvKps(kp[id1]), ArrayFromCvKps(kp[id2]), inlier_matches)
            plt.figure(figsize=(12, 12))
            plt.imshow(im)
            plt.title(f"{pair}: R err={err_q:.2f}°, T err={err_t:.2f}m")
            plt.axis("off")
            plt.savefig(f"tmp/{pair}.png")

    return errors

