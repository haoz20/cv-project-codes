# Garuda 3D Reconstruction — CV Pipeline Sample Code

Our own OpenCV/NumPy code for the pipeline in
[`garuda_cv_pipeline.md`](garuda_cv_pipeline.md), in two parts:

- **Stage 2 — Feature Detection & Matching** (`src/01`–`03`). Stages 3–4 (pose
  estimation and sparse SfM) run through COLMAP.
- **Stages 5–6 — novel-view rendering** (`src/04`–`06`), the Mac-native Ch. 14 track
  described in [`PLAN.md`](PLAN.md). Two renderers, both running on the M2 Air in
  minutes with no CUDA and no training:
  - `06_view_interpolation.py` — **the one to use.** View interpolation (§14.1):
    morphs between adjacent photos through dense correspondence. Sharp, full-frame,
    sweeps the whole arc.
  - `04`+`05` — view-dependent texture mapping through a single plane proxy (§14.1.1,
    §14.3.1). Kept as the baseline the interpolation result is measured against; `06`
    still uses the plane `04` fits.

## Setup

Use the Anaconda interpreter — it has `opencv-python`, `numpy`, and `matplotlib` installed:

```bash
/opt/anaconda3/bin/python --version   # should be 3.12.7
```

No `pip install` needed; nothing beyond `opencv-python`, `numpy`, `matplotlib` is used.

## Data

- `data/sample/` — three photos of a desk scene (img1–img3.jpg), the placeholder input the
  Stage 2 scripts default to. Not representative subject matter, but real multi-view photos
  with real overlap.
- `data/garuda/` — empty by default; the museum photos now live under `colmap_work/images/`
  (below). Point the Stage 2 scripts at them with `--images colmap_work/images/` — no code
  changes needed.
- `colmap_work/` — the real museum capture and its reconstruction (git-ignored; unpack
  `garuda_colmap_data.zip`). `images/` holds the 99 source photos, `sparse/0/` the COLMAP
  sparse model: 97/99 images registered, 64,945 points, 1.37 px mean reprojection error.
  Stages 5–6 read this directly and need nothing else.

## Scripts

Full copy-paste command reference, including the real museum photo folders and the demo pair:
[`COMMANDS.md`](COMMANDS.md).

Run everything from the repo root. All output lands in `output/` (git-ignored).

### Stage 2 — feature detection and matching

```bash
/opt/anaconda3/bin/python src/01_detect_features.py
```

```bash
/opt/anaconda3/bin/python src/02_match_pairs.py
```

```bash
/opt/anaconda3/bin/python src/03_verify_matches.py
```

Each accepts:
- `--images <folder>` — override the input folder (default `data/sample`)
- `--show` — also pop up the matplotlib figure window, in addition to saving it

### Stage 6b — view interpolation (the good one)

Needs the plane from `04` as its base warp, so run that first:

```bash
/opt/anaconda3/bin/python src/04_plane_proxy.py
```

```bash
/opt/anaconda3/bin/python src/06_view_interpolation.py --pingpong
```

Writes `output/garuda_view_interpolation.mp4` and
`output/06_view_interpolation_frames.png`. Takes about a minute.

### Stages 5–6 — view-dependent texture mapping (baseline)

Two steps, in order — `05` needs the plane `04` writes. Both default to
`colmap_work/`, so no arguments are needed:

```bash
/opt/anaconda3/bin/python src/04_plane_proxy.py
```

```bash
/opt/anaconda3/bin/python src/05_view_dependent_render.py --pingpong
```

`04` takes a few seconds; `05` takes a couple of minutes, most of it spent decoding the
8064×6048 source JPEGs. Between them they write:

| File | What it is |
| --- | --- |
| `output/plane_proxy.json` | the fitted plane, read by `05` |
| `output/04_plane_proxy.png` | fit diagnostics (edge-on view + residual histogram) |
| `output/garuda_view_dependent_sweep.mp4` | the plane-proxy arc sweep (baseline) |
| `output/05_view_dependent_frames.png` | start/middle/end frames, for the report |

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

### `04_plane_proxy.py`
Fits the dominant plane that Stage 6 projects through, using the sparse 3D points from
`colmap_work/sparse/0`. RANSAC first (robust against background and floor points), then a
least-squares SVD refit through the whole subject — RANSAC on its own latches onto the flat
wall *behind* the relief, leaving the carving entirely in front of the proxy. The refit
halves the depth error that shows up as ghosting later (RMS residual 0.28 → 0.17 COLMAP
units, for a 3.5° change in orientation).

Prints a sanity check against the capture geometry, as a wrong plane is easy to miss
otherwise: all 97 cameras should sit in front of the plane and face it. They do — 100% in
front, optical axes a median 37° off the plane normal.

Useful flags: `--refit-band 0` (disable the refit and keep the raw RANSAC plane),
`--threshold` (RANSAC inlier distance; defaults to 2% of the scene scale),
`--min-track` / `--max-error` (how aggressively to filter the input points).

### `05_view_dependent_render.py`
The Stage 6 deliverable. For each virtual camera along the arc it warps the nearest source
photos into that view through the plane homography
`H = K_v (R_vi + t_vi n_iᵀ / d_i) K_i⁻¹`, weights them per pixel by the angle between the
virtual view ray and each source camera's ray to the same point on the proxy, and blends.
Weights follow the unstructured-lumigraph form `w = max(0, 1/θ − 1/θ_k)` — plain `1/θ`
(Debevec et al. 1996) makes photos pop in and out as they enter and leave the k-nearest set,
and subtracting the furthest selected photo's weight decays each contribution to zero right
before it is dropped.

Two things worth knowing about the output:

- **It sweeps the middle of the arc, not all of it.** The photos span 167° around the
  subject, which puts the outermost cameras nearly edge-on to the plane, where a planar
  proxy projects to a sliver and the frame degenerates into extreme skew. `--arc-limit`
  (default 55°) bounds the *path* to where the proxy holds, while still blending from every
  photo, including those outside that range. Raise it to see the failure mode.
- **Some blur is inherent.** A single plane cannot represent the relief's ~0.3-unit depth,
  so photos taken from different angles disagree slightly and the blend softens. That is the
  cost of skipping mesh reconstruction — worth naming directly in the Results comparison
  against the NeRF track rather than presenting the sweep as artefact-free.

Useful flags: `--sources` (photos blended per frame; fewer is sharper but pops more),
`--frames` / `--fps` / `--width` (video size), `--max-dim` (source photo resolution),
`--pingpong` (sweep out and back, so the video loops seamlessly), `--arc-limit`.

### `06_view_interpolation.py`
Renders the sweep by **view interpolation** (Szeliski §14.1, Chen and Williams 1993)
instead of through a geometric proxy. It takes adjacent photos along the arc, builds a
dense pixel-to-pixel correspondence between them, and morphs one into the other to
synthesise the frames in between. Correspondence is found "plane + parallax": the plane
from `04` supplies a homography that removes everything the flat wall explains, and dense
optical flow (DIS) picks up the small residual, which is exactly the parallax from the
relief standing off that wall. Running flow on the residual rather than the raw pair is
what makes it reliable.

This is the renderer to use. Because correspondence is per pixel, depth is handled
implicitly and exactly, so the ghosting that the single plane cannot avoid simply does not
arise — alignment error between adjacent photos drops 2–4× (mean abs. error 15.7 → 3.9
grey levels at a typical 6.5° step, 34.5 → 17.9 at the worst 10.6° step). And because
every frame is anchored to real photographs rather than a synthetic viewpoint, there are
no black wedges and no arc limit: the full sweep renders, not just the middle.

Two things it works out from the data:

- **Adjacency comes from shooting order**, not COLMAP image ID and not angle around the
  arc. Consecutive shots in a handheld walk-around are consecutive in space (median 6.8°
  apart); image-ID order gives 13°, and arc-angle order 13.9°, because two photos can
  share an arc angle while being metres apart in height.
- **The capture is four passes, not one.** The sequence is cut wherever consecutive photos
  jump more than `--max-step` (default 12°), which finds four sweeps — two of them
  complete, 28 photos over ~170°. `--sweep 1` renders the second one.

Useful flags: `--frames` / `--fps` / `--max-dim` (size and length), `--sweep`,
`--pingpong`, `--no-stabilise` (keep the raw handheld framing), `--zoom`.

## Recommended before the museum visit

Do a dry-run capture: 15–20 photos circling any object at home, following the Stage 1 capture
protocol (60–80% overlap, fixed focal length, consistent lighting). Drop them in `data/garuda/`
and re-run `02_match_pairs.py --images data/garuda/`. This validates the capture protocol and
gives far more representative numbers than the desk photos, without needing a second museum trip
if something's off.
