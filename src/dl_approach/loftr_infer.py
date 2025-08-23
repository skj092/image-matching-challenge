import pandas as pd
import os
from pathlib import Path
import cv2
import torch
from kornia_moons.viz import draw_LAF_matches
import kornia as K
import kornia.feature as KF


def load_torch_image(fname, device):
    img = cv2.imread(fname)
    scale = 840 / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() / 255.0
    img = K.color.bgr_to_rgb(img)
    return img.to(device)


data_dir = Path("data/train/taj_mahal")
df = pd.read_csv(os.path.join(data_dir, "pair_covisibility.csv"))

pair = df.iloc[0]["pair"]
img1, img2 = pair.split("-")

img1 = os.path.join(data_dir, "images", img1 + ".jpg")
img2 = os.path.join(data_dir, "images", img2 + ".jpg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matcher = KF.LoFTR(pretrained=None)
matcher.load_state_dict(torch.load("models/loftr_outdoor.ckpt")["state_dict"])
matcher = matcher.to(device).eval()


image1 = load_torch_image(img1, device)
image2 = load_torch_image(img2, device)

input_dict = {
    "image0": K.color.rgb_to_grayscale(image1),
    "image1": K.color.rgb_to_grayscale(image2),
}

with torch.no_grad():
    output_dict = matcher(input_dict)

mkpts0 = output_dict["keypoints0"].cpu().numpy()
mkpts1 = output_dict["keypoints1"].cpu().numpy()

F, inliers = cv2.findFundamentalMat(
    mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.1845, 0.999999, 220000
)
inliers = inliers > 0
assert F.shape == (3, 3), "Malformed F?"


fig, ax = draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(image1),
    K.tensor_to_image(image2),
    inliers,
    draw_dict={
        "inlier_color": (0.2, 1, 0.2),
        "tentative_color": None,
        "feature_color": (0.2, 0.5, 1),
        "vertical": False,
    },
    return_fig_ax=True,
)


fig.savefig("tmp/matches.jpg")
print("tmp/matches.jpg saved")
