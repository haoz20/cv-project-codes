"""
Stage 2 — Feature Detection & Matching
2. Feature Matching: SIFT + BFMatcher (Lowe's ratio test) vs ORB + Hamming

Based on Szeliski's "Computer Vision: Algorithms and Applications", Ch. 7,
and the same approach used in the Ch. 8 / Week 6 coursework
(chapter8/ex5-ch8.py, Week6/ex1_w6.py).

Matches the first two images in the input folder and reports, for both
SIFT and ORB:
  - keypoint counts
  - raw knn matches vs "good" matches after Lowe's ratio test
  - a sweep of the ratio threshold (0.6 / 0.7 / 0.75 / 0.8), to justify
    the 0.75 threshold used elsewhere rather than treating it as a
    magic number
  - wall-clock time per detector

This table is the core Results content for the progress check.
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import bgr2rgb, load_folder, make_arg_parser, save_figure, timed

RATIO_THRESHOLDS = [0.6, 0.7, 0.75, 0.8]


def ratio_test(knn_matches, ratio):
    good = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def match_sift(gray1, gray2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good = ratio_test(knn_matches, 0.75)
    return kp1, kp2, knn_matches, good


def match_orb(gray1, gray2, nfeatures=2000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good = ratio_test(knn_matches, 0.75)
    return kp1, kp2, knn_matches, good


def draw_top_matches(img1, kp1, img2, kp2, matches, n=60):
    matches_sorted = sorted(matches, key=lambda m: m.distance)[:n]
    return cv2.drawMatches(
        img1, kp1, img2, kp2, matches_sorted, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def main():
    parser = make_arg_parser("Match a pair of images with SIFT vs ORB and report statistics.")
    args = parser.parse_args()

    images = load_folder(args.images)
    if len(images) < 2:
        raise SystemExit(f"Need at least 2 images in {args.images}, found {len(images)}")

    (name1, img1), (name2, img2) = images[0], images[1]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    print(f"Matching pair: {name1}  <->  {name2}\n")

    (kp1_s, kp2_s, knn_s, good_s), t_sift = timed(match_sift, gray1, gray2)
    (kp1_o, kp2_o, knn_o, good_o), t_orb = timed(match_orb, gray1, gray2)

    print(f"{'Detector':<10}{'#KP img1':<10}{'#KP img2':<10}{'#Raw':<8}{'#Good(0.75)':<13}{'Time (s)':<10}")
    print(f"{'SIFT':<10}{len(kp1_s):<10}{len(kp2_s):<10}{len(knn_s):<8}{len(good_s):<13}{t_sift:<10.3f}")
    print(f"{'ORB':<10}{len(kp1_o):<10}{len(kp2_o):<10}{len(knn_o):<8}{len(good_o):<13}{t_orb:<10.3f}")

    print(f"\nLowe's ratio threshold sweep (SIFT, {name1} <-> {name2}):")
    print(f"{'Ratio':<8}{'#Good matches':<15}")
    for ratio in RATIO_THRESHOLDS:
        good = ratio_test(knn_s, ratio)
        print(f"{ratio:<8}{len(good):<15}")

    vis_sift = draw_top_matches(img1, kp1_s, img2, kp2_s, good_s)
    plt.figure(figsize=(14, 6))
    plt.title(f"SIFT matches (ratio test 0.75, {len(good_s)} good) — {name1} vs {name2}")
    plt.imshow(bgr2rgb(vis_sift))
    plt.axis("off")
    save_figure("02_matches_sift", show=args.show)

    vis_orb = draw_top_matches(img1, kp1_o, img2, kp2_o, good_o)
    plt.figure(figsize=(14, 6))
    plt.title(f"ORB matches (ratio test 0.75, {len(good_o)} good) — {name1} vs {name2}")
    plt.imshow(bgr2rgb(vis_orb))
    plt.axis("off")
    save_figure("02_matches_orb", show=args.show)


if __name__ == "__main__":
    main()
