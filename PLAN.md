# PLAN — View-Dependent Texture Mapping (Mac-native Ch.14 backup)

**Why this exists:** the primary Ch.14 track (NeRF via Instant-NGP on the Windows GTX 1650
laptop) has real open risk — unverified `colmap2nerf.py` flags, unconfirmed whether 4GB VRAM is
enough for this dataset. This is a parallel, Mac-only backup: no CUDA, no borrowed hardware, no
training loop, runs entirely on the M2 Air with what's already built. Not a replacement — build
this alongside Instant-NGP, not instead of it, and use whichever finishes first / works better.

**Technique:** View-Dependent Texture Mapping (Szeliski §14.1.1) / Unstructured Lumigraph
Rendering (§14.3.1) — Debevec, Taylor, and Malik (1996). Blend the original source photos at
render time, weighted by angular closeness to the virtual camera, projected through a simple
geometric proxy. On the Ch.14 taxonomy as a "hybrid geometry + image-based" pipeline — the same
category Photo Tourism (Snavely, Seitz, and Szeliski 2006) uses for exactly this reason: no mesh
reconstruction needed, no MVS, no training.

---

## Inputs (already exist, nothing new required)

- `colmap_work/images/` — 99 source photos
- `colmap_work/sparse/0/` — COLMAP sparse model: camera poses/intrinsics + 3D point cloud
  (97/99 registered, 64,945 points, 1.37px mean reprojection error)

## Stages

### 1–4. Image acquisition → feature matching → pose estimation → sparse SfM
Already done, unchanged from the main pipeline (`garuda_cv_pipeline.md`).

### 5. Plane proxy fitting (new)
Fit a single dominant plane to the sparse 3D point cloud from `colmap_work/sparse/0/points3D`.
This is not a simplification requiring justification — it's the same choice Photo Tourism makes
("a simple dominant plane fit to the 3D points visible in each image often performs better,"
per the book), and it fits this subject better than it fits theirs: the Garuda is a wall-mounted
relief, genuinely close to planar, not a free-standing object needing a full mesh.

- Load `points3D.bin`/`.txt` (COLMAP binary or text format, already exported alongside the
  sparse model).
- Fit via SVD/PCA (centroid + smallest-variance eigenvector as plane normal) or RANSAC plane
  fitting for robustness against outlier points.
- Sanity-check: the fitted plane's normal should roughly align with the capture geometry (camera
  positions should mostly lie on one side, facing the plane) — verify against a few known camera
  poses before trusting it.

### 6. View-dependent rendering (new — the actual deliverable)
For a virtual camera at any pose along the capture arc:

1. Project the virtual camera and every source camera onto the plane proxy.
2. Weight each source photo by angular closeness between the virtual view ray and that source
   camera's ray to the same surface point — weight ∝ 1/angle (Debevec et al. 1996).
3. Warp each weighted source photo onto the virtual view via the plane homography, blend.
4. Repeat across a smooth path of virtual camera poses **swept across the real ~180° capture
   arc** (not a full 360° orbit — the subject has no back to show, and claiming a full orbit
   would misrepresent the capture; sweeping the actual captured arc is the honest and correct
   choice) to produce a video.

Output: a rendered `.mp4` sweep — the primary deliverable for this track, directly comparable
to the NeRF turntable video for a side-by-side Results-section comparison.

## Implementation notes

- Pure NumPy/OpenCV — `cv2.findHomography`, `cv2.warpPerspective`, standard weighted blending.
  No PyTorch, no GPU framework required (though nothing stops using PyTorch on CPU/MPS if it
  simplifies the vectorized blending — not required for correctness or speed at this scale).
- Reuses the existing Anaconda environment (`/opt/anaconda3/bin/python`) — same interpreter as
  `src/01`–`03`, no new environment setup.
- Runtime: seconds to low minutes for the whole render — no training wait, unlike NeRF.

## Deliverables

1. `src/plane_proxy.py` (or similar) — fits and saves the dominant plane from the sparse model.
2. `src/view_dependent_render.py` — renders the arc-sweep video given the plane, poses, and
   source photos.
3. Output video (`output/garuda_view_dependent_sweep.mp4`) for direct comparison against the
   Instant-NGP turntable in the Results section.

## Open questions

- [x] Confirm COLMAP's `points3D` export format/field layout for the plane-fitting step
      (binary vs. text, already have both from the `colmap model_converter` run)
      — **text**. `src/colmap_model.py` reads `cameras.txt` / `images.txt` / `points3D.txt`;
      no struct unpacking needed. Note the model is COLMAP 3.12 format, so `sparse/0` also
      contains `rigs.txt` / `frames.txt`, which are redundant for a single handheld camera
      and are ignored.
- [x] Decide on virtual camera path parameterization for the arc sweep — **hybrid**.
      Centres are interpolated along the real camera centres (smoothed with a 5-camera moving
      average), so the virtual camera stays on the arc that was actually photographed and
      every frame has genuinely nearby source views to blend. Orientations are *not* slerped
      from the source poses — that inherits every framing wobble of a handheld capture —
      but rebuilt as a look-at towards the subject centroid, which keeps the relief centred.
- [ ] If this ends up being the primary deliverable (NeRF track fails), update
      `garuda_cv_pipeline.md`'s Approach line and Stage 5/6 accordingly rather than leaving two
      conflicting "current pipeline" docs
      — *not triggered yet; `garuda_cv_pipeline.md` still describes the NeRF track as primary
      and has deliberately been left alone.*

## Change of approach: view interpolation is the primary renderer

The plane-proxy render (Stage 6 as planned) works but is soft — a single plane cannot
represent the relief's ~0.3-unit depth, so photos taken from different angles disagree and
the blend turns that disagreement into ghosting. `src/06_view_interpolation.py` replaces it
with **view interpolation** (Szeliski §14.1, Chen and Williams 1993), which is still Ch. 14
and still Mac-native, but drops the global geometric model entirely: adjacent photos are
put into dense pixel-to-pixel correspondence and morphed into each other.

Correspondence is "plane + parallax", so the Stage 5 plane fit is still load-bearing — it
supplies the homography that removes everything the flat wall explains, leaving dense
optical flow to pick up only the relief's parallax. Running flow on that small residual
rather than the raw pair is what makes it reliable.

What this buys, measured:

- alignment error between adjacent photos drops 2-4x (mean abs. error 15.7 -> 3.9 grey
  levels at a typical 6.5 deg step; 34.5 -> 17.9 at the worst 10.6 deg step)
- no black wedges — every frame is anchored to real photographs, not a synthetic viewpoint
- no arc limit — the full ~170 deg sweep renders, not just the middle 110 deg

Keep `04`+`05` as the baseline: "single-plane proxy vs. per-pixel correspondence" is a
sharper Results comparison than either render alone, and both are the same chapter.

Two facts about the capture that fell out of building it:

- **Adjacency is shooting order.** Consecutive shots in a handheld walk-around are
  consecutive in space (median 6.8 deg apart). COLMAP image-ID order gives 13 deg, and
  sorting by angle around the arc gives 13.9 deg — two photos can share an arc angle while
  being metres apart in height. Sorting by arc angle was the first thing tried and it put
  non-adjacent views next to each other, which made the flow step fail outright.
- **The capture is four passes across the relief, not one** — two of them complete, 28
  photos over ~170 deg. The renderer cuts the sequence at jumps over `--max-step` and
  sweeps one pass; `--sweep 1` selects the second complete one.

## What the build turned up

Two things the plan did not anticipate, both handled in the scripts:

1. **RANSAC alone fits the wrong plane.** The single thinnest, densest surface in the sparse
   cloud is the flat wall *behind* the relief, so a plain RANSAC fit leaves the entire carving
   in front of the proxy. Stage 5 now refits through the middle of the subject slab, halving
   the depth error that becomes ghosting in the render (RMS 0.28 → 0.17 COLMAP units).
2. **The sweep covers the middle of the arc, not all 167°.** The plan's "sweep the real ~180°
   arc" is right about not faking a 360° orbit, but a *planar* proxy cannot be rendered from
   near-grazing angles — at the arc ends the plane projects to a sliver. `--arc-limit`
   (default ±55°) bounds the path to where the proxy is well conditioned, while still
   blending from every photo. This is a limit of the proxy, not of the fit, and is worth
   stating plainly in the Results section rather than quietly cropping the sweep.
