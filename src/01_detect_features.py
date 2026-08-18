"""
Stage 2 — Feature Detection & Matching
1. Keypoint Detection: SIFT vs ORB

Based on Szeliski's "Computer Vision: Algorithms and Applications", Ch. 7.
Detects keypoints on a single image with both SIFT and ORB and draws them
with scale + orientation ("rich" keypoints) so the difference in coverage
and density between the two detectors is visible in the figure.
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import bgr2rgb, load_folder, make_arg_parser, save_figure, timed


def detect_sift(gray):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)


def detect_orb(gray, nfeatures=2000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    return orb.detectAndCompute(gray, None)


def draw_keypoints(img, keypoints):
    return cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def main():
    parser = make_arg_parser("Detect and visualize SIFT vs ORB keypoints on one image.")
    args = parser.parse_args()

    images = load_folder(args.images)
    name, img = images[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Using image: {name}  ({img.shape[1]}x{img.shape[0]})\n")

    (kp_sift, _), t_sift = timed(detect_sift, gray)
    (kp_orb, _), t_orb = timed(detect_orb, gray)

    print(f"{'Detector':<10}{'#Keypoints':<14}{'Time (s)':<10}")
    print(f"{'SIFT':<10}{len(kp_sift):<14}{t_sift:<10.3f}")
    print(f"{'ORB':<10}{len(kp_orb):<14}{t_orb:<10.3f}")

    sift_vis = draw_keypoints(img, kp_sift)
    plt.figure(figsize=(10, 7))
    plt.title(f"SIFT keypoints ({len(kp_sift)}) — {name}")
    plt.imshow(bgr2rgb(sift_vis))
    plt.axis("off")
    save_figure("01_keypoints_sift", show=args.show)

    orb_vis = draw_keypoints(img, kp_orb)
    plt.figure(figsize=(10, 7))
    plt.title(f"ORB keypoints ({len(kp_orb)}) — {name}")
    plt.imshow(bgr2rgb(orb_vis))
    plt.axis("off")
    save_figure("01_keypoints_orb", show=args.show)


if __name__ == "__main__":
    main()
