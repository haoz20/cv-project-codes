# Command Reference

Copy-paste commands for running the Stage 2 code. All commands run from the repo root:

```bash
cd /Users/znhao/Documents/1-2026/Computer_Vision/term-project/cv-project-codes
```

Always use the Anaconda interpreter — it has `opencv-python`, `numpy`, and `matplotlib`; the
plain `python3` on PATH is a different install missing matplotlib:

```bash
/opt/anaconda3/bin/python --version   # should print 3.12.7
```

---

## Quick reference

```bash
# Detection demo (best pair — dense carved detail, 4178 SIFT keypoints)
/opt/anaconda3/bin/python src/01_detect_features.py --images data/demo --show

# Matching demo (same pair — 2342 good matches, 87% RANSAC inlier ratio)
/opt/anaconda3/bin/python src/02_match_pairs.py --images data/demo --show

# Geometric verification demo (same pair)
/opt/anaconda3/bin/python src/03_verify_matches.py --images data/demo --show
```

Drop `--show` to just save PNGs to `output/` without popping up windows (e.g. for a batch run
before class rather than a live demo).

---

## Flags (all three scripts)

| Flag | Default | Meaning |
|---|---|---|
| `--images <folder>` | `data/sample` | Folder to read photos from. Scripts use the first one or two files, sorted by filename. |
| `--show` | off | Also pop up each matplotlib figure (blocks until you close the window) in addition to saving it. |

---

## Which folder to point `--images` at

| Folder | Contents | Use for |
|---|---|---|
| `data/demo/` | `IMG_6015.JPG`, `IMG_6016.JPG` — the strongest real pair found so far (87% inlier ratio) | **Live demos** — reliable, visually clean, dense keypoints on the face/headdress/wing carving |
| `images/low/` | 20 real photos, low camera angle | Real numbers from the low band |
| `images/mid/` | 28 real photos, eye-level | Real numbers from the mid band (default first pair: `IMG_6003`/`6004`, 84% inlier ratio) |
| `images/high/` | 51 real photos, high angle (avoid `IMG_6108.JPG` — blurry) | Real numbers from the high band |
| `data/sample/` | 3 placeholder desk photos | Fallback only — pre-dates the museum shoot, not the real subject |

Not tracked in git (all four folders above are gitignored) — they only exist on this machine.
See [`PROGRESS_LOG.md`](PROGRESS_LOG.md) for the numbers each folder has already produced.

---

## Script-by-script

### `01_detect_features.py` — keypoint detection

```bash
/opt/anaconda3/bin/python src/01_detect_features.py --images data/demo
```

Uses the **first image** in the folder. Prints a SIFT vs ORB keypoint count/timing table.
Saves `output/01_keypoints_sift.png` and `output/01_keypoints_orb.png`.

### `02_match_pairs.py` — feature matching

```bash
/opt/anaconda3/bin/python src/02_match_pairs.py --images data/demo
```

Uses the **first two images** in the folder. Prints:
- comparison table (keypoints, raw matches, good matches after Lowe's ratio test, timing) for SIFT and ORB
- ratio-threshold sweep (0.6 / 0.7 / 0.75 / 0.8) justifying the 0.75 cutoff

Saves `output/02_matches_sift.png` and `output/02_matches_orb.png`.

### `03_verify_matches.py` — RANSAC geometric verification

```bash
/opt/anaconda3/bin/python src/03_verify_matches.py --images data/demo
```

Uses the **first two images** in the folder. Fits a fundamental matrix with RANSAC to the SIFT
matches from `02`, prints the inlier count/ratio. Saves `output/03_inlier_matches.png`.

Needs at least 8 good matches to run — if you point it at two photos with too little overlap it
exits with an error message rather than crashing.

---

## Running a specific pair (not the first two in a folder)

The scripts always take the first one or two files in `--images <folder>`, sorted by filename.
To demo a specific pair, make (or overwrite) a small folder with just those files, e.g.:

```bash
mkdir -p data/demo
cp images/mid/IMG_6015.JPG images/mid/IMG_6016.JPG data/demo/
```

Then run any script with `--images data/demo` as above. This is how `data/demo/` was built.

---

## Regenerating a specific `output/` figure

`output/` only ever holds the *last* run of each script — running `02_match_pairs.py` again
overwrites `output/02_matches_sift.png`. If you need to keep a figure, rename or move it before
running a different pair:

```bash
mv output/03_inlier_matches.png output/03_inlier_matches_mid_band.png
/opt/anaconda3/bin/python src/03_verify_matches.py --images images/low --show
```

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Used the wrong interpreter — must be `/opt/anaconda3/bin/python`, not plain `python3`. |
| `FileNotFoundError: No images found in ...` | Wrong `--images` path, or the folder is empty/has no `.jpg`/`.png` files. |
| `Need at least 2 images in ...` | `--images` folder has only 0–1 photos; `02`/`03` need at least two. |
| `03_verify_matches.py` exits with "Only N good matches found" | The chosen pair has too little overlap (< 8 matches) — pick a different pair, e.g. two consecutive photos from the same band rather than across bands. |
| Figure window doesn't appear with `--show` | Matplotlib backend issue — check the script printed `[saved] output/...png` first; the file is still written even if the window fails to pop up. |
