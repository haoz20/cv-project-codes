# CV Pipeline — Garuda 3D Rendering Project (CSX4213)

**Subject:** Garuda statue, Garuda Museum, Samut Prakan
**Approach:** Classical mesh photogrammetry (Szeliski Ch. 13) — feature matching → camera pose
estimation → sparse SfM → dense multi-view stereo → Poisson surface reconstruction → texture →
export. Runs entirely locally across two machines: sparse SfM on the MacBook Air M2 (CPU), dense
reconstruction on the Windows gaming laptop's GTX 1650 (CUDA). No cloud at any stage.
**Team:** Sam Yati, Thiri Htet, Swan Htet Aung — Vincent Mary School of Engineering, Science and Technology, Assumption University

---

## ✅ Coverage concern — resolved 2026-08-28, with real COLMAP data

Lecturer feedback: capture doesn't have enough angle coverage for real 3D reconstruction. A quick
synthetic test (SIFT on downsampled images, 4×4 pairs per boundary) initially looked alarming —
only 7–14 RANSAC inliers per pair at both the mid↔low and low↔high transitions, consistent with
the 8-inlier pair flagged in `PROGRESS_LOG.md` on 2026-08-24.

**But the real COLMAP pipeline (full resolution, 9,000–13,000+ SIFT features/image, exhaustive
matching across all 99 photos) resolved this cleanly:**

```
colmap feature_extractor  → 48 min  (full-res SIFT, all 99 images)
colmap exhaustive_matcher → 5.4 min (4,851 pairs)
colmap mapper             → 2.5 min → ONE unified reconstruction, not split by band

Registered images: 97 / 99   (IMG_6081 and IMG_6082 excluded — not IMG_6108, which registered fine)
Points:            64,945
Mean track length: 5.2
Mean reprojection error: 1.37px  (good — under the usual 2px threshold)
```

The low/mid/high bands **did** unify into a single coherent model — the quick synthetic test
undersold it (fewer features, downsampled images, only 4×4 pairs sampled per boundary vs.
COLMAP's exhaustive search across everything). Video-frame bridging turned out to be unnecessary.
**This is also a stronger rebuttal to the lecturer's concern than argument alone** — cite these
numbers directly rather than the theoretical Ch.13 facade-photogrammetry precedent.

Output in use for Stage 5 onward: `colmap_work/sparse/0` (COLMAP sparse model, poses + 3D points).

---

## Methods considered and rejected

Two neural-rendering approaches were evaluated before settling on the classical pipeline below,
worth recording as part of the design story rather than hiding:

- **Instant-NGP (NeRF)** on the local GTX 1650. Technically ran, but training images loaded
  entirely into VRAM — 97 photos × ~195 MB each ≈ 19 GB against the card's 4 GB — and the
  resolution reduction attempted (`--scale 0.5`) was roughly an order of magnitude too shallow to
  fit. Not a hardware ceiling so much as an under-shrunk input.
- **3D Gaussian Splatting (gsplat)** on the same card. Verified technically workable — gsplat
  ships prebuilt CUDA wheels compatible with the 1650's Turing architecture — but only for Python
  3.10 on Windows, forcing a parallel install alongside the laptop's Python 3.13 plus exact
  torch/CUDA version pinning. The classical route below reaches a comparably real 3D result
  (actual recovered geometry, not just Instant-NGP's implicit field) with dramatically less
  toolchain risk: COLMAP's own Windows binary ships CUDA precompiled, needing nothing beyond
  unzip-and-run.

Both are legitimate state-of-the-art methods and worth a paragraph in the paper's Methodology as
"considered and rejected for toolchain/hardware reliability on the available machines," rather
than a dead end to omit.

---

## Pipeline

### 1. Image Acquisition — done
- Circular multi-view sweep, capped at ~180° (wall-mounted relief, no back access).
- 60–80% overlap between consecutive photos.
- 3 height levels (low: 20, mid: 28, high: 51 — 99 photos total).
- Consistent focal length and lighting throughout the shoot.

### 2. Feature Detection & Matching
- SIFT (standard, more accurate) or ORB (faster, less accurate).
- `cv2.BFMatcher` + Lowe's ratio test; RANSAC fundamental-matrix verification.
- Own OpenCV code (`src/01`–`03`) — demonstrates the stage directly for the report.

### 3. Camera Pose Estimation
- Estimate essential/fundamental matrix between matched pairs.
- Recover each camera's relative rotation and translation.

### 4. Structure from Motion (SfM) — sparse only — done, 2026-08-28
- Ran in `colmap_work/`: `feature_extractor` (full-res SIFT, `SIMPLE_RADIAL` camera model) →
  `exhaustive_matcher` → `mapper`. Result: **one unified sparse model**, 97/99 images registered,
  64,945 points, mean reprojection error 1.37px. See the resolved coverage note above for the
  full numbers.
- This sparse model (`colmap_work/sparse/0`) is exactly the input Stage 5's dense reconstruction
  consumes — COLMAP never re-runs feature extraction/matching/mapping downstream of this.

### 5. Dense Multi-View Stereo
**Goal:** densify the sparse point cloud using the poses already recovered in Stage 4, via
`colmap patch_match_stereo` on the **local GTX 1650**. Runs from COLMAP's official prebuilt CUDA
Windows binary (`colmap-x64-windows-cuda.zip`) — no compiler, no `nvcc`, no Python version
pinning, unlike either rejected neural approach.

Dense stereo is architecturally lighter on VRAM than neural training: it processes one reference
image against its neighbours at a time rather than holding all cameras and an optimizable
parameter tensor in memory simultaneously. `--PatchMatchStereo.max_image_size` caps the working
resolution (default 2400 in this pipeline's tooling) independent of the full-resolution source
images on disk, and the pipeline auto-retries at a lower cap on a detected CUDA out-of-memory
failure.

**Implementation:** `src/09_dense_stereo.py` (wraps `patch_match_stereo` with the OOM
auto-retry), preceded by `src/07_prepare_scene.py` (scene bundling) and `src/08_undistort.py`
(`image_undistorter`, converts the `SIMPLE_RADIAL` model to the pinhole format dense stereo
needs). Full reference: [docs/dense_pipeline.md](docs/dense_pipeline.md).

### 6. Surface Reconstruction & Texture
- `colmap stereo_fusion` fuses per-view depth maps into a dense, coloured point cloud
  (`fused.ply`) — colour sampled directly from the source photos, order of magnitude denser than
  the 64,945-point sparse model.
- `colmap poisson_mesher` converts that into a triangle mesh, propagating per-vertex colour from
  the fused cloud — a complete, coloured mesh with no additional tool required.
- **Optional stretch:** OpenMVS's `TextureMesh` (also a prebuilt CUDA Windows binary,
  `OpenMVS_Windows_x64_CUDA.7z`) for a proper UV-texture atlas instead of per-vertex colour, if the
  Poisson result looks too coarse for the paper's figures.
- **Implementation:** `src/10_fuse_mesh.py` (fusion + Poisson, plus the point/vertex/face-count
  stats for the Results section), `src/11_texture_openmvs.py` (optional, marked best-effort).

**Expect a hole or noise above the eaves** — every elevation band is still an eye-level-and-up
walkaround; nothing looked straight down onto the top of the relief. Report this as a
capture-coverage limitation, not a failure.

### 7. Export & Rendering
- `src/12_export_mesh.py` converts the final `.ply`/`.obj` to `.glb` (via trimesh) — a single
  self-contained, web-viewable submission format, with the source `.ply`/`.obj` kept as a
  raw/native backup.
- `src/13_turntable_render.py` renders a full 360° orbit to MP4 via Open3D — a genuine upside of
  having recovered real geometry: unlike the earlier image-based-rendering baseline
  (`src/05_view_dependent_render.py`), whose virtual camera had to stay near the captured ~170°
  arc, the camera path here is unconstrained.
- Both stages are light enough to run on the Mac once the mesh exists — the Windows laptop's job
  ends after Stage 6.

**Showcase package for the presentation:**
1. Turntable render video (`.mp4`) — primary deliverable, no live dependency.
2. The exported `.glb` opened live in any standard viewer during the presentation, if the room
   setup allows it — screen-recorded backup either way.
3. A couple of still renders next to matching original photos, for a side-by-side quality
   comparison slide in Results.

---

## This Week's Plan

| Day | Task |
|---|---|
| **1** ✅ | ~~Coverage fix~~ — resolved via real COLMAP run instead (see above), no video bridging needed. Sparse SfM (Stage 4) already done: `colmap_work/sparse/0`, 97/99 images, 1.37px reprojection error. |
| **2** ✅ | Windows gaming laptop set up: NVIDIA driver, COLMAP's prebuilt CUDA binary (unzip-and-run, no compiler). See [docs/setup_rog.md](docs/setup_rog.md). |
| **3** | Transfer `garuda_colmap_data.zip` to the Windows laptop, run `python run_dense_pipeline.py --zip garuda_colmap_data.zip --scene scene` through Stage 6 (dense stereo → fusion → Poisson mesh). |
| **4** | Export (`src/12_export_mesh.py`) and render the turntable (`src/13_turntable_render.py`) — can move back to the Mac for these. Optional: OpenMVS texture pass (Stage 11) if the vertex-colour mesh needs a real UV atlas. |
| **5** | Pull renders/figures for Results; sanity-check against known viewpoints; record fusion/mesh stats (`output/10_fuse_mesh.json`) for the Results table. |
| **6–7** | Write up Methodology/Results/Analysis; update paper's Introduction to reflect this single Ch.13 dense-MVS pipeline (not the earlier NeRF/splatting detours); finalize presentation. |

---

## Tooling Plan (MacBook Air M2 for Stages 1–4 + 7, Windows gaming laptop for Stages 5–6)

| Stage(s) | Tool | Notes |
|---|---|---|
| 2 (feature detection/matching) | OpenCV (Python, Anaconda) | Own code — demonstrates the stage directly, reuses Ch. 8 approach |
| 3–4 (pose → sparse SfM) | **COLMAP** (sparse reconstruction only) | Free, open source. `brew install colmap` on Mac. |
| 5–6 (dense MVS → Poisson → texture) | **COLMAP** (prebuilt CUDA binary) on the **Windows gaming laptop (GTX 1650, local)**, optionally **OpenMVS** for texturing | No cloud, no compiler, no Python version pinning — the same reliability reasoning that ruled out the NeRF/splatting routes (see "Methods considered and rejected" above). Full command reference: [docs/dense_pipeline.md](docs/dense_pipeline.md). |
| 7 (export + turntable render) | Python: **trimesh** (→ `.glb`), **Open3D** (offscreen render → `.mp4`) | Light enough for the Mac; see [docs/setup_mac.md](docs/setup_mac.md) for the Open3D headless-rendering gotcha on macOS. |

**Recommended workflow:** Use OpenCV for Stage 2 to write and explain your own feature-matching
code for the report, run COLMAP through sparse SfM only (Stage 4) on the Mac to get camera poses,
then run dense MVS through Poisson meshing (Stages 5–6) on the Windows laptop with COLMAP's CUDA
binary, and finish export + rendering (Stage 7) back on the Mac. No cloud infrastructure at any
stage. Full pipeline reference, exact commands, and troubleshooting: [docs/](docs/README.md).

---

## Open Questions / To Confirm

- [ ] Confirm OpenMVS's exact `TextureMesh`/`InterfaceCOLMAP` flag names on the installed release
      if the optional UV-texture stretch stage is used — its CLI has changed across releases and
      `src/11_texture_openmvs.py` is marked best-effort rather than verified against a live install.
- [ ] Confirm final export format for submission: `.glb` (current default, self-contained and
      web-viewable) vs. `.obj`/`.ply` alongside it.
- [ ] Note actual dense-stereo processing time, any `--PatchMatchStereo.max_image_size` OOM
      retries triggered, and mesh quality issues (the expected roof/eaves hole, any other missing
      regions) for the Results/Analysis section — `output/10_fuse_mesh.json` records the
      point/vertex/face counts and timings automatically.
- [ ] Rewrite paper's Introduction ¶2 / Abstract / Lit Review to describe this single Ch.13
      dense-MVS pipeline consistently — earlier drafts described a NeRF (Ch.14) approach that no
      longer matches what's being built; mention Instant-NGP/3D Gaussian Splatting only as the
      "considered and rejected" methods note above, not as the primary method.
