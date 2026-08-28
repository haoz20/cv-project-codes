# CV Pipeline — Garuda 3D Rendering Project (CSX4213)

**Subject:** Garuda statue, Garuda Museum, Samut Prakan
**Approach:** Image-based / neural rendering (Szeliski Ch. 14) — feature matching → camera pose
estimation → SfM (sparse poses only, no MVS/mesh) → NeRF training (Nerfstudio `nerfacto` on a
**free Lightning AI Studio GPU** — no paid cloud, no card on file) → novel-view/turntable
rendering + live interactive viewer. No mesh reconstruction stage; geometry is implicit in the
trained radiance field.
**Team:** Sam Yati, Thiri Htet, Swan Htet Aung — Vincent Mary School of Engineering, Science and Technology, Assumption University

---

## ⚠️ Blocking item — do this before anything else this week

Lecturer feedback: capture doesn't have enough angle coverage for real 3D reconstruction. This
lines up with a real gap already flagged in `PROGRESS_LOG.md` — the low↔high band boundary had
only **8 RANSAC inliers** on the one pair tested, the weakest link in the whole capture. NeRF
training needs enough parallax/overlap between neighboring viewpoints to be supervised correctly,
so this has to be fixed before Stage 4 (SfM) is trusted.

**No return museum visit is possible — fix this from the walkthrough video shot the same day
instead of reshooting.**

- [ ] **Locate the walkthrough video(s)** from the 2026-08-22 visit and identify which
      timestamp ranges cover the low→mid and mid→high transitions (the camera moving/tilting
      between height bands).
- [ ] **Extract candidate frames from those segments** with ffmpeg, oversampling (e.g. ~3–5 fps)
      so there's a selection to filter from:
      ```bash
      ffmpeg -i walkthrough.mp4 -vf "select='between(t,START,END)'" -vsync vfr \
        -q:v 2 bridge_frames/low_mid_%04d.jpg
      ```
      Run once per transition segment (adjust `START`/`END` per range identified above).
- [ ] **Filter out blurry/motion-blurred frames** — handheld video frames are much more prone to
      motion blur than the deliberate stills. Use a Laplacian-variance sharpness check (reuse or
      extend `src/common.py`) and keep only frames above a sharpness threshold; expect to keep a
      minority of extracted frames.
- [ ] **Fold surviving frames into the same flattened image folder** used for COLMAP's
      `exhaustive_matcher` (per the 2026-08-24 decision — don't re-split into bands). Treat them
      as a separate camera model in COLMAP if the video was shot in a different capture mode than
      the stills (different crop/FOV) — don't assume identical intrinsics to the photo set.
- [ ] **Re-check the boundary specifically** before trusting the full run: re-run
      `src/03_verify_matches.py` on a low-band photo vs. a nearby new bridge frame, and a bridge
      frame vs. a high-band photo, to confirm the inlier count actually improved past the
      original 8 before spending Day 2–3 on the full SfM/NeRF runs.

---

## Pipeline

### 1. Image Acquisition
- Circular multi-view sweep, capped at ~180° (wall-mounted relief, no back access).
- 60–80% overlap between consecutive photos; densified at height-band transitions via the
  video-extracted bridging frames above.
- 2–3 height levels (low, eye-level, above).
- Consistent focal length and lighting throughout the shoot.
- ~100–120 frames total (100 original stills + filtered video bridging frames).

### 2. Feature Detection & Matching
- SIFT (standard, more accurate) or ORB (faster, less accurate).
- `cv2.BFMatcher` + Lowe's ratio test; RANSAC fundamental-matrix verification.
- Own OpenCV code (`src/01`–`03`) — demonstrates the stage directly for the report, and is what's
  used to sanity-check the bridging frames above before committing to a full run.

### 3. Camera Pose Estimation
- Estimate essential/fundamental matrix between matched pairs.
- Recover each camera's relative rotation and translation.

### 4. Structure from Motion (SfM) — sparse only
- COLMAP `exhaustive_matcher` (not sequential — the low/mid/high bands are separate numbering
  blocks with only a thin bridge between them) registers all cameras into one shared coordinate
  frame and triangulates a **sparse** 3D point cloud, refined via bundle adjustment.
- **Stop here** — this pipeline does not run COLMAP's dense MVS, Poisson surface reconstruction,
  or texture mapping. The only output needed downstream is the sparse point cloud + per-image
  camera poses/intrinsics, which is exactly what Stage 5 (NeRF) consumes.

### 5. Neural Radiance Field Training (Ch. 14 core)
**Goal:** train an actual image-based-rendering model — a NeRF — on the posed photo set, using
Nerfstudio's `nerfacto` on a **free Lightning AI Studio GPU** (no paid cloud — decided against
OCI/Modal to avoid any risk of charges or a card on file).

**B1. Lightning AI Studio (primary), Colab/Kaggle backup**
- **Lightning AI Studio (free tier)** — 80 GPU-hours/month (15 credits), T4 16GB, **no credit
  card required**. Persistent VM with a real terminal (not a notebook kernel) — storage survives
  between sessions, and you can `tmux`/background a training run the same way the original
  OCI plan assumed.
- **Colab / Kaggle** — backup only, if Lightning's free GPU queue is unavailable or the monthly
  80 hours run out. Notebook-based, storage resets on disconnect (see earlier notes if falling
  back to this).

**B2. Install & data prep (Studio terminal)**
```bash
pip install nerfstudio
```
- Upload the flattened image folder + Stage 4's COLMAP sparse model into the Studio's persistent
  storage (drag-and-drop in the UI, or `scp`/`rsync` if Lightning exposes SSH for your plan).
```bash
ns-process-data images --data ~/garuda/images --output-dir ~/garuda/processed \
  --colmap-model-path ~/garuda/colmap_sparse
```

**B3. Train (Studio terminal)**
```bash
tmux new -s nerf   # survive disconnects, same as the original plan
ns-train nerfacto --data ~/garuda/processed
```
- ~15–30 min on a T4 for a scene this size.
- Capture is a limited-angle (~180°) sweep, not a full orbit — this is fine for NeRF: treat it as
  a **forward-facing scene** (the same regime as the original NeRF paper's LLFF dataset), not an
  unbounded 360° scene. `nerfacto` handles this by default; if results look warped, check
  `--pipeline.model.disable-scene-contraction`.
- Free-tier GPU time runs on interruptible machines — some preemption risk, same category as
  Colab's disconnects. Storage persists regardless, so a preempted/resumed run doesn't lose data,
  but check training progress periodically rather than assuming a long unattended run completes.

**B4. Live viewer — genuinely available here, unlike Colab**
- In the Studio UI: **Web Apps** tab (right sidebar) → **Port Viewer** → add port `7007`.
- Lightning prints a public `*.hrzn.run` URL — open it to get Nerfstudio's actual interactive
  viewer (rotate/pan/zoom the trained Garuda live), during training or after loading a finished
  checkpoint via `ns-viewer --load-config outputs/.../config.yml`.
- **For the presentation itself:** don't rely on live wifi + this URL in the room — screen-record
  yourself navigating the live viewer ahead of time as a safety-net video, and have the `hrzn.run`
  link ready as a live bonus if the room's internet cooperates.

### 6. Novel View Synthesis / Rendering
- Render a turntable/rotation path through the trained NeRF — the primary deliverable,
  demonstrating actual image-based rendering (viewpoint-dependent synthesis) rather than a static
  export. Build the camera path by clicking keyframes in the live viewer (B4) while orbiting the
  model, then export it:
```bash
ns-render camera-path --load-config outputs/.../config.yml \
  --camera-path-filename <path.json> --output-path renders/garuda_turntable.mp4
```
- Optional: `ns-export poisson --load-config outputs/.../config.yml --output-dir exports/` to
  pull a mesh out of the trained field, only if a mesh file is required for submission format —
  this is a byproduct of the NeRF, not a separate reconstruction pipeline.

**Showcase package for the presentation:**
1. Turntable render video (`.mp4`) — safe primary, no live dependency
2. Screen-recorded clip of live-navigating the viewer via the `hrzn.run` URL — shows it's a real
   interactive model, not just a canned video
3. A couple of still renders next to matching original photos, for a side-by-side quality
   comparison slide in Results

---

## This Week's Plan

| Day | Task |
|---|---|
| **1** | Extract + filter bridging frames from the walkthrough video; verify boundary inlier count improved; flatten into one folder. Sign up for Lightning AI Studio (no card), install Nerfstudio in a Studio terminal so it's ready to go. |
| **2** | Run COLMAP `exhaustive_matcher` → sparse SfM (Stage 4). |
| **3** | Upload data to the Studio, `ns-process-data` with Stage 4's COLMAP poses, start `ns-train nerfacto` in `tmux`. Expose port 7007 via Port Viewer and screen-record a live navigation clip once training looks good. |
| **4** | Build a camera path in the live viewer, render turntable video (+ optional mesh export). |
| **5** | Pull renders/figures for Results; sanity-check against known viewpoints. |
| **6–7** | Write up Methodology/Results/Analysis; update paper's Introduction to reflect this single Ch.14 pipeline (not the earlier mesh-pipeline draft); finalize presentation. |

---

## Tooling Plan (MacBook Air M2, 16GB + free-tier cloud GPU)

| Stage(s) | Tool | Notes |
|---|---|---|
| 2 (feature detection/matching) | OpenCV (Python, Anaconda) | Own code — demonstrates the stage directly, reuses Ch. 8 approach |
| 3–4 (pose → sparse SfM) | **COLMAP** (sparse reconstruction only) | Free, open source. `brew install colmap`. Stop after sparse SfM — no dense MVS needed for this pipeline. |
| 5–6 (NeRF training → rendering + live viewer) | **Nerfstudio (`nerfacto`)** on **Lightning AI Studio (free tier)** | 80 GPU-hrs/month, T4, **no card required**. Persistent terminal (not a notebook), built-in Port Viewer for the live interactive viewer. Backup: Colab/Kaggle if the free GPU queue is unavailable. Ruled out: OCI (quota/approval risk, card on file), Modal (free tier effectively needs a card for real usage, serverless model doesn't fit an interactive live-viewer workflow). |

**Recommended workflow:** Use OpenCV for Stage 2 to write and explain your own feature-matching
code for the report, run COLMAP through sparse SfM only (Stage 4) to get camera poses, then train
and render the NeRF (Stages 5–6) on Lightning AI Studio, using its Port Viewer to get the live
interactive demo. No mesh reconstruction stage in this pipeline, and no paid cloud infrastructure.

---

## Open Questions / To Confirm

- [ ] Locate the exact walkthrough video file(s) and note their timestamp ranges/duration —
      needed before the ffmpeg extraction step above can run
- [ ] Confirm video-extracted bridging frames actually close the low↔mid/mid↔high gap (re-check
      inlier counts on those pairs before trusting the full exhaustive match)
- [ ] Confirm Lightning AI Studio's free GPU queue is available when needed (interruptible
      machines — some preemption risk); fall back to Colab/Kaggle rather than losing Day 3 if not
- [ ] Confirm submission format: turntable video only, or does a mesh export (via `ns-export`)
      also need to be included?
- [ ] Note actual processing time and any failure points (NeRF floaters, artifacts, blurry
      regions from the sparse-coverage side) for the Results/Analysis section
- [ ] Rewrite paper's Introduction ¶2 / Abstract / Lit Review to describe this single Ch.14
      NeRF pipeline consistently — the earlier drafts committed to a Ch.13 mesh pipeline that no
      longer matches what's being built
