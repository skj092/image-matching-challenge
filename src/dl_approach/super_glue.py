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

# Load data
data_dir = Path("data/train/taj_mahal")
df = pd.read_csv(os.path.join(data_dir, "pair_covisibility.csv"))

# Pick first pair
pair = df.iloc[0]["pair"]
img1_name, img2_name = pair.split("-")

# Image paths
img1_path = os.path.join(data_dir, "images", img1_name + ".jpg")
img2_path = os.path.join(data_dir, "images", img2_name + ".jpg")


device = "cuda" if torch.cuda.is_available() else "cpu"
resize = [
    -1,
]
resize_float = True

config = {
    "superpoint": {"nms_radius": 4, "keypoint_threshold": 0.005, "max_keypoints": 1024},
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2,
    },
}
matching = Matching(config).eval().to(device)
image_1, inp_1, scales_1 = read_image(img1_path, device, resize, 0, resize_float)
image_2, inp_2, scales_2 = read_image(img2_path, device, resize, 0, resize_float)

pred = matching({"image0": inp_1, "image1": inp_2})
pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
kpts1, kpts2 = pred["keypoints0"], pred["keypoints1"]
matches, conf = pred["matches0"], pred["matching_scores0"]

valid = matches > -1
mkpts1 = kpts1[valid]
mkpts2 = kpts2[matches[valid]]
mconf = conf[valid]
F, inlier_mask = cv2.findFundamentalMat(
    mkpts1,
    mkpts2,
    cv2.USAC_MAGSAC,
    ransacReprojThreshold=0.25,
    confidence=0.99999,
    maxIters=10000,
)

# Plotting function from SuperGlue utils
color = cm.jet(mconf)  # colormap for match confidence
text = [f"Image pair: {img1_name} - {img2_name}", f"Matches: {len(mkpts1)}"]

out = make_matching_plot(
    image_1,
    image_2,
    kpts1,
    kpts2,
    mkpts1,
    mkpts2,
    color,
    text,
    path="tmp/superglue.jpg",
    show_keypoints=True,
    fast_viz=False,
    opencv_display=False,
    opencv_title="Matches",
)
