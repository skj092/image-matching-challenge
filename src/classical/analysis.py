import os
from collections import namedtuple

import cv2
import numpy as np

from utils import load_scaling_factors, process_scene

Gt = namedtuple("Gt", ["K", "R", "T"])
eps = 1e-15

# ----------------------------------------------------
# Main
# ----------------------------------------------------


def main():
    src = "data/train"
    os.makedirs("tmp", exist_ok=True)

    sift = cv2.SIFT_create(5000, contrastThreshold=-10000, edgeThreshold=-10000)
    scaling_dict = load_scaling_factors(f"{src}/scaling_factors.csv")

    all_errors = {}
    for scene in scaling_dict.keys():
        print(f"\n--- Processing {scene} ---")
        errors = process_scene(scene, src, sift, scaling_dict, show_matches=True)
        all_errors[scene] = errors

        rq = [v[0] for v in errors.values()]
        rt = [v[1] for v in errors.values()]
        print(f"Scene {scene}: Avg Rot={np.mean(rq):.2f}°, Avg Trans={np.mean(rt):.2f}m")

    all_q = [v[0] for errs in all_errors.values() for v in errs.values()]
    all_t = [v[1] for errs in all_errors.values() for v in errs.values()]
    print("\n=== Overall ===")
    print(f"Avg Rotation Error: {np.mean(all_q):.2f}°")
    print(f"Avg Translation Error: {np.mean(all_t):.2f}m")


if __name__ == "__main__":
    main()

