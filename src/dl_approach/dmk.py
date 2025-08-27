from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import pandas as pd
from pathlib import Path

sys.path.append("external/dkm")
from dkm.utils.utils import tensor_to_pil

from dkm import DKMv3_outdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_dir = Path("data/train/taj_mahal")
df = pd.read_csv(os.path.join(data_dir, "pair_covisibility.csv"))

pair = df.iloc[0]["pair"]
img1, img2 = pair.split("-")

img1 = os.path.join(data_dir, "images", img1 + ".jpg")
img2 = os.path.join(data_dir, "images", img2 + ".jpg")


dkm_model = DKMv3_outdoor(device=device)

H, W = 864, 1152

im1 = Image.open(img1).resize((W, H))
im2 = Image.open(img2).resize((W, H))

# Match
warp, certainty = dkm_model.match(im1, im2, device=device)
# Sampling not needed, but can be done with model.sample(warp, certainty)
dkm_model.sample(warp, certainty)
x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

im2_transfer_rgb = F.grid_sample(
    x2[None], warp[:, :W, 2:][None], mode="bilinear", align_corners=False
)[0]
im1_transfer_rgb = F.grid_sample(
    x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
)[0]
warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb), dim=2)
white_im = torch.ones((H, 2 * W), device=device)
vis_im = certainty * warp_im + (1 - certainty) * white_im
tensor_to_pil(vis_im, unnormalize=False).save("tmp/dmk.jpg")
