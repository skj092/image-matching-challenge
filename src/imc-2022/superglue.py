from pathlib import Path
import os
import pandas as pd
import cv2
from matplotlib import cm
import torch
import sys

# https://github.com/magicleap/SuperGluePretrainedNetwork
sys.path.append("external/SuperGluePretrainedNetwork")
from models.matching import Matching
from models.utils import make_matching_plot, read_image


src = '/kaggle/input/image-matching-challenge-2022/'

test_samples = []
with open(f'{src}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]
test_samples_df = pd.DataFrame(test_samples, columns=["sample_id", "batch_id", "image_1_id", "image_2_id"])

device = "cuda" if torch.cuda.is_available() else "cpu"
resize = [-1, ]
resize_float = True

config = {
    "superpoint": {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": 1024
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2,
    }
}
matching = Matching(config).eval().to(device)

F_dict = {}
for i, row in tqdm(enumerate(test_samples)):
    sample_id, batch_id, image_1_id, image_2_id = row

    image_fpath_1 = f'{src}/test_images/{batch_id}/{image_1_id}.png'
    image_fpath_2 = f'{src}/test_images/{batch_id}/{image_2_id}.png'

    image_1, inp_1, scales_1 = read_image(image_fpath_1, device, resize, 0, resize_float)
    image_2, inp_2, scales_2 = read_image(image_fpath_2, device, resize, 0, resize_float)

    pred = matching({"image0": inp_1, "image1": inp_2})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts1, kpts2 = pred["keypoints0"], pred["keypoints1"]
    matches, conf = pred["matches0"], pred["matching_scores0"]

    valid = matches > -1
    mkpts1 = kpts1[valid]
    mkpts2 = kpts2[matches[valid]]
    mconf = conf[valid]

    if len(mkpts1) > 8:
        F, inlier_mask = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.USAC_MAGSAC, ransacReprojThreshold=0.25, confidence=0.99999, maxIters=10000)
        F_dict[sample_id] = F
    else:
        F_dict[sample_id] = np.zeros((3, 3))

def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])

with open('submission.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in F_dict.items():
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')
