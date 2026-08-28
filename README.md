# Garuda 3D Reconstruction — CV Pipeline Sample Code

Stage 2 (Feature Detection & Matching) of the pipeline described in
[`garuda_cv_pipeline.md`](garuda_cv_pipeline.md). This is the stage handled with
our own OpenCV code (Stages 3–7 will run through COLMAP once the full photo set exists).

## Setup

Use the Anaconda interpreter — it has `opencv-python`, `numpy`, and `matplotlib` installed:

```bash
/opt/anaconda3/bin/python --version   # should be 3.12.7
```

No `pip install` needed; nothing beyond `opencv-python`, `numpy`, `matplotlib` is used.

## Data

- `data/sample/` — three photos of a desk scene (img1–img3.jpg), used as placeholder input
  until the actual Garuda photos are captured. Not representative subject matter, but real
  multi-view photos with real overlap, so the pipeline produces real numbers today.
- `data/garuda/` — empty. Drop the museum photo set here once captured; every script accepts
  `--images data/garuda/` and needs no code changes.

## Scripts

Full copy-paste command reference, including the real museum photo folders and the demo pair:
[`COMMANDS.md`](COMMANDS.md).

Run from the repo root:

```bash
/opt/anaconda3/bin/python src/01_detect_features.py
/opt/anaconda3/bin/python src/02_match_pairs.py
/opt/anaconda3/bin/python src/03_verify_matches.py
```

Each accepts:
- `--images <folder>` — override the input folder (default `data/sample`)
- `--show` — also pop up the matplotlib figure window, in addition to saving it

All figures are written to `output/` (git-ignored) as PNGs, ready to paste into the report.

### `01_detect_features.py`
Detects keypoints on the first image in the folder with **SIFT** and **ORB**, prints keypoint
counts and detection time, and saves both keypoint visualizations.

### `02_match_pairs.py`
Matches the first two images in the folder with **SIFT + BFMatcher (Lowe's ratio test)** and
**ORB + Hamming distance**. Prints a comparison table (keypoints / raw matches / good matches /
time) and a sweep of the ratio threshold (0.6 / 0.7 / 0.75 / 0.8) justifying the 0.75 cutoff.
Saves the top matches for both detectors. **This is the main Results-section output.**

### `03_verify_matches.py`
Fits a fundamental matrix with RANSAC to the SIFT matches from `02` and reports the inlier
ratio — matches that survive geometric verification, vs. raw ratio-test matches. Bridges into
Stage 3 (Camera Pose Estimation). Optional: drop this file for a strictly Stage-2-only
submission, since `01` and `02` don't depend on it.

## Recommended before the museum visit

Do a dry-run capture: 15–20 photos circling any object at home, following the Stage 1 capture
protocol (60–80% overlap, fixed focal length, consistent lighting). Drop them in `data/garuda/`
and re-run `02_match_pairs.py --images data/garuda/`. This validates the capture protocol and
gives far more representative numbers than the desk photos, without needing a second museum trip
if something's off.
