"""
Stage 2 — Feature Detection & Matching
3. Geometric Verification: Fundamental Matrix + RANSAC

Based on Szeliski's "Computer Vision: Algorithms and Applications", Ch. 7.

Raw "good" matches from Lowe's ratio test still contain outliers that are
geometrically inconsistent between the two views. This script estimates the
fundamental matrix between a matched pair with RANSAC and reports the
inlier ratio — a more honest match-quality number than raw good-match
counts, and the natural bridge into Stage 3 (Camera Pose Estimation).

Note on scope: garuda_cv_pipeline.md lists the fundamental matrix under
Stage 3. It is used here purely as a geometric filter on Stage 2 matches,
not to recover camera pose/rotation/translation. Drop this file if a
strictly Stage-2-only submission is wanted; 01 and 02 do not depend on it.
"""

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import bgr2rgb, load_folder, make_arg_parser, save_figure, timed

RATIO = 0.75


def match_sift(gray1, gray2):
    """
    Same SIFT + ratio-test matching as 02_match_pairs.py.
    Duplicated (rather than imported across a numeric-prefixed module) so
    this file runs standalone — same pattern as chapter8/ex5-ch8.py.
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn_matches:
        if m.distance < RATIO * n.distance:
            good.append(m)
    return kp1, kp2, good


def verify_with_fundamental_matrix(kp1, kp2, good_matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if mask is None:
        return F, np.zeros(len(good_matches), dtype=bool)
    return F, mask.ravel().astype(bool)


def main():
    parser = make_arg_parser("Verify SIFT matches with a RANSAC fundamental-matrix fit.")
    args = parser.parse_args()

    images = load_folder(args.images)
    if len(images) < 2:
        raise SystemExit(f"Need at least 2 images in {args.images}, found {len(images)}")

    (name1, img1), (name2, img2) = images[0], images[1]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    print(f"Verifying pair: {name1}  <->  {name2}\n")

    (kp1, kp2, good), t_match = timed(match_sift, gray1, gray2)

    if len(good) < 8:
        raise SystemExit(
            f"Only {len(good)} good matches found — need at least 8 to fit a "
            "fundamental matrix. Try a pair with more overlap."
        )

    (F, inlier_mask), t_ransac = timed(verify_with_fundamental_matrix, kp1, kp2, good)
    inliers = [m for m, keep in zip(good, inlier_mask) if keep]

    inlier_ratio = len(inliers) / len(good)
    print(f"{'#Good matches':<16}{'#Inliers':<12}{'Inlier ratio':<14}{'Match time (s)':<16}{'RANSAC time (s)':<16}")
    print(f"{len(good):<16}{len(inliers):<12}{inlier_ratio:<14.3f}{t_match:<16.3f}{t_ransac:<16.3f}")

    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, inliers, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.figure(figsize=(14, 6))
    plt.title(
        f"RANSAC inlier matches ({len(inliers)}/{len(good)}, "
        f"ratio {inlier_ratio:.2f}) — {name1} vs {name2}"
    )
    plt.imshow(bgr2rgb(vis))
    plt.axis("off")
    save_figure("03_inlier_matches", show=args.show)


if __name__ == "__main__":
    main()
