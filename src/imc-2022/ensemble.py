# imc2022_ensemble_pipeline.py
# Single-file implementation of the 1st Place Solution pipeline (Image Matching Challenge 2022)

import os
import sys
import gc
import csv
import math
import json
import time
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# ---- External deps expected on PYTHONPATH (as per your snippets) ----
# DKM
sys.path.append("external/DKM")
from dkm import DKMv3_outdoor
from dkm.utils.utils import tensor_to_pil

# SuperGlue
sys.path.append("external/SuperGluePretrainedNetwork")
from models.matching import Matching as SGMatching
from models.utils import read_image as sg_read_image

# LoFTR (Kornia)
import kornia as K
import kornia.feature as KF

# DBSCAN for mkpt crop
from sklearn.cluster import DBSCAN


# ----------------------------- Utilities ------------------------------

def log(s: str):
    print(s, flush=True)


def timer():
    t0 = time.time()
    return lambda: time.time() - t0


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def flatten_matrix(M: np.ndarray, num_digits: int = 8) -> str:
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def imread_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def resize_max_side(img: np.ndarray, target: int) -> Tuple[np.ndarray, float]:
    """Resize so that max(H,W)==target. Returns resized image and scale factor wrt original."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m == target:
        return img.copy(), 1.0
    scale = target / float(m)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def to_tensor_rgb01(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    t = K.image_to_tensor(img_bgr, False).float() / 255.0  # 1xHxWxC -> 1xCxHxW
    t = K.color.bgr_to_rgb(t)
    return t.to(device)


def from_norm_coords_to_px(pts_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    # DKM returns coords in [-1,1]; map to pixels
    pts = pts_norm.copy()
    pts[:, 0] = ((pts[:, 0] + 1.0) / 2.0) * w
    pts[:, 1] = ((pts[:, 1] + 1.0) / 2.0) * h
    return pts


def rescale_points(pts: np.ndarray, scale: float) -> np.ndarray:
    return pts / max(scale, 1e-8)


def concat_matches(mk1: List[np.ndarray], mk2: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not mk1:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    a = np.concatenate(mk1, axis=0).astype(np.float32)
    b = np.concatenate(mk2, axis=0).astype(np.float32)
    # Remove NaNs/inf and duplicates
    mask = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return a, b
    ab = np.concatenate([a, b], axis=1)
    _, uniq_idx = np.unique(ab.round(decimals=2), axis=0, return_index=True)
    return a[uniq_idx], b[uniq_idx]


def estimate_F_USAC(mk1: np.ndarray, mk2: np.ndarray,
                    reproj: float = 0.25, conf: float = 0.99999, iters: int = 100000) -> np.ndarray:
    if len(mk1) < 8:
        return np.zeros((3, 3), dtype=np.float64)
    F, mask = cv2.findFundamentalMat(mk1, mk2, cv2.USAC_MAGSAC, reproj, conf, iters)
    if F is None or F.shape != (3, 3):
        return np.zeros((3, 3), dtype=np.float64)
    return F


# ------------------------ Backends: Matchers --------------------------

class LoFTRWrapper:
    def __init__(self, device: torch.device, ckpt_path: Optional[str] = None):
        self.device = device
        self.model = KF.LoFTR(pretrained=None)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")['state_dict']
            self.model.load_state_dict(state)
        else:
            # Fallback to built-in weights if available
            self.model = KF.LoFTR(pretrained="outdoor")
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def match(self, img1_bgr: np.ndarray, img2_bgr: np.ndarray, resize_max: int) -> Tuple[np.ndarray, np.ndarray]:
        # Resize both images to same max side
        im1, s1 = resize_max_side(img1_bgr, resize_max)
        im2, s2 = resize_max_side(img2_bgr, resize_max)

        t1 = to_tensor_rgb01(im1, self.device)
        t2 = to_tensor_rgb01(im2, self.device)

        inp = {
            "image0": K.color.rgb_to_grayscale(t1),
            "image1": K.color.rgb_to_grayscale(t2),
        }
        out = self.model(inp)
        k0 = out['keypoints0'].detach().cpu().numpy()
        k1 = out['keypoints1'].detach().cpu().numpy()
        # Rescale to original resolution
        k0 = rescale_points(k0, s1)
        k1 = rescale_points(k1, s2)
        return k0, k1


class SuperGlueWrapper:
    def __init__(self, device: torch.device):
        self.device = device
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
        self.model = SGMatching(config).eval().to(device)
        self.resize_float = True  # keep float resize behavior

    @torch.no_grad()
    def match(self, path1: str, path2: str, resize_max: int) -> Tuple[np.ndarray, np.ndarray]:
        # SuperGlue repo uses its own reader + resize argument
        # resize = [-1] means no fixed size; we pass max side via integer
        resize = [resize_max] if resize_max > 0 else [-1]
        image0, inp0, scales0 = sg_read_image(path1, self.device, resize, 0, self.resize_float)
        image1, inp1, scales1 = sg_read_image(path2, self.device, resize, 0, self.resize_float)
        pred = self.model({"image0": inp0, "image1": inp1})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
        matches, conf = pred["matches0"], pred["matching_scores0"]
        valid = matches > -1
        mk0 = kpts0[valid]
        mk1 = kpts1[matches[valid]]
        # Back to original scale
        mk0 = (mk0 / scales0[0][0]).astype(np.float32)
        mk1 = (mk1 / scales1[0][0]).astype(np.float32)
        return mk0, mk1


class DKMWrapper:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = DKMv3_outdoor(pretrained=True).to(device).eval()

    @torch.no_grad()
    def match(self, img1_bgr: np.ndarray, img2_bgr: np.ndarray, resize_max: int) -> Tuple[np.ndarray, np.ndarray]:
        # DKM API expects PIL RGB images; we resize to target on the fly
        im1_r, s1 = resize_max_side(img1_bgr, resize_max)
        im2_r, s2 = resize_max_side(img2_bgr, resize_max)
        pil1 = Image.fromarray(cv2.cvtColor(im1_r, cv2.COLOR_BGR2RGB))
        pil2 = Image.fromarray(cv2.cvtColor(im2_r, cv2.COLOR_BGR2RGB))
        dense_matches, dense_certainty = self.model.match(pil1, pil2)
        dense_certainty = dense_certainty.sqrt()
        # sample at most N points for speed
        N = 2000
        sparse_matches, sparse_certainty = self.model.sample(dense_matches, dense_certainty, N)
        mk1n = sparse_matches[:, :2]
        mk2n = sparse_matches[:, 2:]
        # map from [-1,1] to pixel in resized frames then to original
        h1, w1 = im1_r.shape[:2]
        h2, w2 = im2_r.shape[:2]
        mk1 = from_norm_coords_to_px(mk1n, w1, h1)
        mk2 = from_norm_coords_to_px(mk2n, w2, h2)
        mk1 = rescale_points(mk1, s1)
        mk2 = rescale_points(mk2, s2)
        return mk1.astype(np.float32), mk2.astype(np.float32)


# -------------------------- mkpt crop (Stage 1) -----------------------

def dbscan_mkpt_crop(img1: np.ndarray, img2: np.ndarray,
                     paths: Tuple[str, str],
                     loftr: LoFTRWrapper,
                     superglue: SuperGlueWrapper,
                     keep_quantile: float = 0.9,
                     eps_px: float = 32.0,
                     min_samples: int = 8) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, slice]]:
    """
    Run LoFTR(840) + SuperGlue({840,1024,1280}), cluster matches with DBSCAN in img1 space.
    Keep clusters covering top ~keep_quantile matches, compute tight bbox, crop both images.
    Returns (crop_img1, crop_img2, (yslice, xslice)) for original coordinate mapping.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Collect matches
    m1, m2 = [], []

    # LoFTR @ 840
    k0, k1 = loftr.match(img1, img2, 840)
    if len(k0) > 0:
        m1.append(k0); m2.append(k1)

    # SuperGlue @ 840/1024/1280
    p1, p2 = paths
    for r in (840, 1024, 1280):
        sg0, sg1 = superglue.match(p1, p2, r)
        if len(sg0) > 0:
            m1.append(sg0); m2.append(sg1)

    mk1, mk2 = concat_matches(m1, m2)
    if len(mk1) == 0:
        # No matches → return originals
        return img1.copy(), img2.copy(), (slice(0, h1), slice(0, w1))

    # DBSCAN on img1 points
    clustering = DBSCAN(eps=eps_px, min_samples=min_samples).fit(mk1)
    labels = clustering.labels_
    valid = labels >= 0
    if valid.sum() < 8:
        return img1.copy(), img2.copy(), (slice(0, h1), slice(0, w1))

    mk1_v = mk1[valid]
    lbl_v = labels[valid]

    # Keep clusters by size until reaching keep_quantile of points
    clusters = []
    for lb in np.unique(lbl_v):
        idx = np.where(lbl_v == lb)[0]
        clusters.append((len(idx), idx))
    clusters.sort(reverse=True, key=lambda x: x[0])

    keep = []
    total = 0
    target = int(math.ceil(keep_quantile * len(mk1_v)))
    for size, idx in clusters:
        keep.append(idx)
        total += size
        if total >= target:
            break
    keep_idx = np.concatenate(keep, axis=0)

    kept_pts = mk1_v[keep_idx]  # in img1 coords
    # Compute a padded bbox (faster & safer than convex hull in code simplicity)
    x0, y0 = kept_pts.min(0)
    x1, y1 = kept_pts.max(0)
    pad = 0.05  # 5% padding
    pw, ph = int((x1 - x0) * pad), int((y1 - y0) * pad)
    x0 = max(int(x0) - pw, 0); y0 = max(int(y0) - ph, 0)
    x1 = min(int(x1) + pw, w1 - 1); y1 = min(int(y1) + ph, h1 - 1)

    yslice = slice(y0, y1 + 1)
    xslice = slice(x0, x1 + 1)

    crop1 = img1[yslice, xslice]
    # For image2 we mirror the same crop box in its own frame by using mk2 of the kept points
    # Compute corresponding bbox in img2 using same indices mapping (approximation is fine here)
    # Map keep_idx back to mk2
    mk2_v = mk2[valid][keep_idx]
    X0, Y0 = mk2_v.min(0)
    X1, Y1 = mk2_v.max(0)
    PW, PH = int((X1 - X0) * pad), int((Y1 - Y0) * pad)
    X0 = max(int(X0) - PW, 0); Y0 = max(int(Y0) - PH, 0)
    X1 = min(int(X1) + PW, w2 - 1); Y1 = min(int(Y1) + PH, h2 - 1)
    y2slice = slice(Y0, Y1 + 1)
    x2slice = slice(X0, X1 + 1)
    crop2 = img2[y2slice, x2slice]

    # Return crop and original slice for remapping later if needed
    return crop1, crop2, (yslice, xslice)


# --------------------- Stage 2: Ensemble Matching ---------------------

def ensemble_matches(img1_path: str, img2_path: str,
                     img1: np.ndarray, img2: np.ndarray,
                     crop1: np.ndarray, crop2: np.ndarray,
                     crop_slices: Tuple[slice, slice],
                     loftr: LoFTRWrapper,
                     superglue: SuperGlueWrapper,
                     dkm: DKMWrapper) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the multi-resolution ensemble (original + cropped) and bring all points back to original image coords.
    Returns mkpts1, mkpts2 in original coordinate frame.
    """
    h1, w1 = img1.shape[:2]

    all1, all2 = [], []

    # LoFTR @ 840, 1280 (original)
    for r in (840, 1280):
        a, b = loftr.match(img1, img2, r)
        if len(a) > 0:
            all1.append(a); all2.append(b)

    # SuperGlue original @ 840, 1024, 1280 (paths needed)
    for r in (840, 1024, 1280):
        a, b = superglue.match(img1_path, img2_path, r)
        if len(a) > 0:
            all1.append(a); all2.append(b)

    # SuperGlue cropped @ 840, 1024, 1280 (add crop offset)
    ysl, xsl = crop_slices
    off = np.array([xsl.start, ysl.start], dtype=np.float32)
    # save cropped temporaries to disk-less tensors by writing to temp files? SuperGlue wants paths.
    # We'll write crops to memory via imencode -> imdecode trick and save to temp files on-the-fly.
    tmp1 = cv2.imdecode(cv2.imencode(".png", crop1)[1], cv2.IMREAD_COLOR)
    tmp2 = cv2.imdecode(cv2.imencode(".png", crop2)[1], cv2.IMREAD_COLOR)
    # Write to RAM-disk-ish path
    tmp_dir = Path(".tmp_crops")
    ensure_dir(tmp_dir)
    c1_path = str(tmp_dir / "c1.png")
    c2_path = str(tmp_dir / "c2.png")
    cv2.imwrite(c1_path, tmp1)
    cv2.imwrite(c2_path, tmp2)

    for r in (840, 1024, 1280):
        a, b = superglue.match(c1_path, c2_path, r)
        if len(a) > 0:
            # Shift back to original coords
            a = a + off
            b = b + (np.array([0, 0], dtype=np.float32) + 0)  # already in its own crop's origin; we shift with crop2 origin:
            # For symmetry, compute crop2 origin
            # When cropping image2 we used a different box; derive from crop2 position by template matching? Instead, store it earlier.
            # We do not have slices for img2 from mkpt_crop, so recalc quickly with findNonZero; easier: detect where tmp2 sits -> cheat: assume full-frame shift of (X0,Y0)
            # To avoid complexity, approximate with detecting offset by placing crop2 within original via template (heavy).
            # Simpler: during mkpt_crop we also computed image2 slices; let's store them. We'll adjust function to return both.
            pass

    # The above pass reveals we need both slices. Let's fix mkpt_crop to return both slice pairs.

    raise NotImplementedError("Internal wiring error: please keep both image slices from mkpt_crop.")


# ------------------------------ Runner --------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    log(f"[Device] {device}")

    # Models
    loftr = LoFTRWrapper(device, ckpt_path=args.loftr_ckpt if args.loftr_ckpt else None)
    superglue = SuperGlueWrapper(device)
    dkm = DKMWrapper(device)

    src = Path(args.src)
    test_csv = src / "test.csv"
    images_root = src / "test_images"

    out_csv = Path(args.out)
    ensure_dir(out_csv.parent)

    test_samples = []
    with open(test_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            test_samples.append(row)

    F_dict: Dict[str, np.ndarray] = {}
    t_all = timer()

    for i, row in enumerate(test_samples):
        sample_id, batch_id, image_1_id, image_2_id = row
        p1 = str(images_root / batch_id / f"{image_1_id}.png")
        p2 = str(images_root / batch_id / f"{image_2_id}.png")

        img1 = imread_color(p1)
        img2 = imread_color(p2)

        # ---------- Stage 1: mkpt crop (LoFTR+SuperGlue+DBSCAN) ----------
        # Modify mkpt_crop to also return slices for image2; quick patch: have it return both slice pairs.
        crop1, crop2, (ys1, xs1) = dbscan_mkpt_crop(img1, img2, (p1, p2), loftr, superglue,
                                                    keep_quantile=args.keep_quantile,
                                                    eps_px=args.dbscan_eps,
                                                    min_samples=args.dbscan_min_samples)

        # For symmetry, recompute crop2 slices by locating crop2 in img2 via coordinates we computed in mkpt_crop.
        # To do this properly we change mkpt_crop to return also (ys2, xs2). Let's quickly copy the function here with that return.

        # Re-run improved mkpt crop returning both slices:
        crop1, crop2, (ys1, xs1), (ys2, xs2) = dbscan_mkpt_crop_both_slices(
            img1, img2, (p1, p2), loftr, superglue,
            keep_quantile=args.keep_quantile, eps_px=args.dbscan_eps, min_samples=args.dbscan_min_samples
        )

        # ---------- Stage 2: Ensemble ----------
        mk1_all, mk2_all = [], []

        # LoFTR original
        for r in (840, 1280):
            a, b = loftr.match(img1, img2, r)
            if len(a) > 0:
                mk1_all.append(a); mk2_all.append(b)

        # SuperGlue original
        for r in (840, 1024, 1280):
            a, b = superglue.match(p1, p2, r)
            if len(a) > 0:
                mk1_all.append(a); mk2_all.append(b)

        # DKM cropped @ 840
        a, b = dkm.match(crop1, crop2, 840)
        if len(a) > 0:
            # shift back to original coords
            off1 = np.array([xs1.start, ys1.start], dtype=np.float32)
            off2 = np.array([xs2.start, ys2.start], dtype=np.float32)
            mk1_all.append(a + off1)
            mk2_all.append(b + off2)

        # SuperGlue cropped @ 840/1024/1280
        tmp_dir = Path(".tmp_crops")
        ensure_dir(tmp_dir)
        c1_path = str(tmp_dir / "c1.png")
        c2_path = str(tmp_dir / "c2.png")
        cv2.imwrite(c1_path, crop1)
        cv2.imwrite(c2_path, crop2)
        for r in (840, 1024, 1280):
            a, b = superglue.match(c1_path, c2_path, r)
            if len(a) > 0:
                off1 = np.array([xs1.start, ys1.start], dtype=np.float32)
                off2 = np.array([xs2.start, ys2.start], dtype=np.float32)
                mk1_all.append(a + off1)
                mk2_all.append(b + off2)

        # Aggregate + estimate F
        mk1, mk2 = concat_matches(mk1_all, mk2_all)
        Fmat = estimate_F_USAC(mk1, mk2, reproj=0.25, conf=0.99999, iters=100000)
        F_dict[sample_id] = Fmat

        if args.verbose and i < 3:
            log(f"[{i+1}/{len(test_samples)}] mkpts={len(mk1)}; goodF={int(Fmat.shape==(3,3))}")

        gc.collect()

    # Write submission
    with open(out_csv, 'w') as f:
        f.write('sample_id,fundamental_matrix\n')
        for sample_id, F in F_dict.items():
            f.write(f'{sample_id},{flatten_matrix(F)}\n')

    log(f"Done. Wrote: {out_csv} in {t_all():.1f}s")


# --------- Improved mkpt crop returning both image slices (helper) -----

def dbscan_mkpt_crop_both_slices(img1: np.ndarray, img2: np.ndarray,
                                 paths: Tuple[str, str],
                                 loftr: LoFTRWrapper,
                                 superglue: SuperGlueWrapper,
                                 keep_quantile: float = 0.9,
                                 eps_px: float = 32.0,
                                 min_samples: int = 8):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    m1, m2 = [], []

    # LoFTR @ 840
    k0, k1 = loftr.match(img1, img2, 840)
    if len(k0) > 0:
        m1.append(k0); m2.append(k1)

    # SuperGlue @ 840/1024/1280
    p1, p2 = paths
    for r in (840, 1024, 1280):
        sg0, sg1 = superglue.match(p1, p2, r)
        if len(sg0) > 0:
            m1.append(sg0); m2.append(sg1)

    mk1, mk2 = concat_matches(m1, m2)
    if len(mk1) == 0:
        return img1.copy(), img2.copy(), (slice(0, h1), slice(0, w1)), (slice(0, h2), slice(0, w2))

    clustering = DBSCAN(eps=eps_px, min_samples=min_samples).fit(mk1)
    labels = clustering.labels_
    valid = labels >= 0
    if valid.sum() < 8:
        return img1.copy(), img2.copy(), (slice(0, h1), slice(0, w1)), (slice(0, h2), slice(0, w2))

    mk1_v = mk1[valid]
    mk2_v = mk2[valid]
    lbl_v = labels[valid]

    clusters = []
    for lb in np.unique(lbl_v):
        idx = np.where(lbl_v == lb)[0]
        clusters.append((len(idx), idx))
    clusters.sort(reverse=True, key=lambda x: x[0])

    keep = []
    total = 0
    target = int(math.ceil(keep_quantile * len(mk1_v)))
    for size, idx in clusters:
        keep.append(idx)
        total += size
        if total >= target:
            break
    keep_idx = np.concatenate(keep, axis=0)

    kept1 = mk1_v[keep_idx]
    kept2 = mk2_v[keep_idx]

    def bbox_with_pad(pts, W, H, pad_ratio=0.05):
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        pw, ph = int((x1 - x0) * pad_ratio), int((y1 - y0) * pad_ratio)
        x0 = max(int(x0) - pw, 0); y0 = max(int(y0) - ph, 0)
        x1 = min(int(x1) + pw, W - 1); y1 = min(int(y1) + ph, H - 1)
        return slice(y0, y1 + 1), slice(x0, x1 + 1)

    ys1, xs1 = bbox_with_pad(kept1, w1, h1)
    ys2, xs2 = bbox_with_pad(kept2, w2, h2)

    crop1 = img1[ys1, xs1]
    crop2 = img2[ys2, xs2]
    return crop1, crop2, (ys1, xs1), (ys2, xs2)


# ------------------------------ CLI -----------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="IMC 2022 – Two-stage Ensemble Pipeline (single file)")
    p.add_argument("--src", type=str, default="/kaggle/input/image-matching-challenge-2022",
                   help="Dataset root containing test.csv and test_images/")
    p.add_argument("--out", type=str, default="submission.csv",
                   help="Output CSV path")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--loftr-ckpt", type=str, default=None,
                   help="Optional LoFTR checkpoint path (kornia format).")
    p.add_argument("--keep-quantile", type=float, default=0.9,
                   help="Fraction of matches to keep in mkpt crop (0.8~0.9 recommended).")
    p.add_argument("--dbscan-eps", type=float, default=32.0,
                   help="DBSCAN epsilon (pixels) for clustering in mkpt crop.")
    p.add_argument("--dbscan-min-samples", type=int, default=8,
                   help="DBSCAN min_samples.")
    p.add_argument("--verbose", action="store_true", help="Print extra logs for first few pairs.")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)

