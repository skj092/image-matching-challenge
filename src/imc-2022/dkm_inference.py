from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import pandas as pd
from pathlib import Path
import csv

sys.path.append("external/DKM")
from dkm.utils.utils import tensor_to_pil

from dkm import DKMv3_outdoor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dry_run = False


def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''

    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


src = '/kaggle/input/image-matching-challenge-2022/'

test_samples = []
with open(f'{src}/test.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]

if dry_run:
    for sample in test_samples:
        print(sample)


F_dict = {}
for i, row in enumerate(test_samples):
    sample_id, batch_id, image_1_id, image_2_id = row

    img1 = cv2.imread(f'{src}/test_images/{batch_id}/{image_1_id}.png')
    img2 = cv2.imread(f'{src}/test_images/{batch_id}/{image_2_id}.png')

    img1PIL = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2PIL = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    dense_matches, dense_certainty = model.match(img1PIL, img2PIL)
    dense_certainty = dense_certainty.sqrt()
    sparse_matches, sparse_certainty = model.sample(dense_matches, dense_certainty, 2000)

    mkps1 = sparse_matches[:, :2]
    mkps2 = sparse_matches[:, 2:]

    h, w, c = img1.shape
    mkps1[:, 0] = ((mkps1[:, 0] + 1)/2) * w
    mkps1[:, 1] = ((mkps1[:, 1] + 1)/2) * h

    h, w, c = img2.shape
    mkps2[:, 0] = ((mkps2[:, 0] + 1)/2) * w
    mkps2[:, 1] = ((mkps2[:, 1] + 1)/2) * h

    F, mask = cv2.findFundamentalMat(mkps1, mkps2, cv2.USAC_MAGSAC, 0.3, 0.9999, 25_000)


    good = F is not None and F.shape == (3,3)

    if good:
        F_dict[sample_id] = F
    else:
        F_dict[sample_id] = np.zeros((3, 3))
        continue

    gc.collect()

with open('submission.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in F_dict.items():
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')

if dry_run:
    !cat submission.csv
