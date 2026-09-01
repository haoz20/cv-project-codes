# Dense pipeline reference

Classical mesh photogrammetry, picking up where the existing sparse SfM
model leaves off:

```
99 photos → COLMAP SfM → sparse/0 → dense MVS → fused.ply → Poisson → mesh → texture → export
   (shot)     (DONE —      97 poses+   (patch_match_   (dense       (mesher)  (vertex   (.obj/
              reuse as-is) 64,945 pts) stereo, GPU)     colored                color, or  .glb)
                                                          cloud)                OpenMVS)
```

Stages 1–4 of the original team plan (capture, features, pose, SfM) are
already done — `colmap_work/sparse/0/` — and are reused unchanged. COLMAP
never re-runs feature extraction/matching/mapping for this track. What
follows is stages 5–8 (renumbered 7–13 here to sit after this repo's
existing `01`–`06` feature/IBR scripts).

## Prerequisites

- **COLMAP**, CUDA build, on the machine running Stages 9 (and 8/10 if
  convenient) — see [setup_rog.md](setup_rog.md).
- **Python deps** — `pip install -r requirements.txt` (trimesh, open3d,
  opencv, numpy, matplotlib). See [setup_mac.md](setup_mac.md).
- **OpenMVS** *(optional, Stage 11 only)* — see [setup_rog.md](setup_rog.md).

## Stage 7 — Prepare the scene

```bash
python src/07_prepare_scene.py --zip garuda_colmap_data.zip --out scene
# or, from local folders instead of the zip:
python src/07_prepare_scene.py --images images --sparse colmap_work/sparse/0 --out scene
```

Builds `scene/images/` + `scene/sparse/0/` from either a
`garuda_colmap_data.zip`-style archive or existing local folders.

**Flattens automatically.** The real `images/` folder is split into
`high/low/mid` subfolders, but the sparse model's `images.txt` was built
against a flat namespace (no subfolder prefix — `colmap_work/images/` is 99
symlinks sitting directly in one folder). `07_prepare_scene.py` walks
`--images` recursively and writes everything flat into `scene/images/`, so
image names line up with the registered poses either way.

**Does not downscale.** Unlike the abandoned 3D Gaussian Splatting plan,
dense MVS and texture quality both benefit from full resolution — COLMAP
caps its own *working* resolution internally (Stage 9's
`--PatchMatchStereo.max_image_size`), not via pre-shrunk files on disk.

**Validated against real data**: on this repo's actual `images/` +
`colmap_work/sparse/0/`, this reports 99 images on disk, 97 registered
poses, 64,945 points — exactly the known-good reconstruction. A mismatch
beyond ±5% on either count prints a warning rather than failing silently.

Output: `scene/images/`, `scene/sparse/0/`, `output/07_prepare_scene.json`

## Stage 8 — Undistort

```bash
python src/08_undistort.py --scene scene
```

Wraps `colmap image_undistorter`. Converts the `SIMPLE_RADIAL` model
(`k1=0.031786`) into undistorted pinhole images + a re-formatted sparse
model, which dense stereo requires. CPU/disk-bound — runs fine on either
machine.

Output: `scene/dense/images/`, `scene/dense/sparse/`, `output/08_undistort.json`

## Stage 9 — Dense stereo (the one GPU/VRAM-relevant stage)

```bash
python src/09_dense_stereo.py --scene scene --max-image-size 2400
```

Wraps `colmap patch_match_stereo`. This is architecturally nothing like 3D
Gaussian Splatting training: it processes one reference image against its
neighbours at a time, never holding all 97 cameras and a giant optimizable
parameter tensor in VRAM simultaneously, so the working set per step is
inherently much smaller.

The one lever that matters at this photo size is
`--PatchMatchStereo.max_image_size` — the resolution *used internally* for
matching, independent of the on-disk image size. At native 8064×6048 this
can get tight on 4 GB; **2400 is the default here**, following COLMAP's own
guidance for smaller-VRAM GPUs.

**Automatic OOM retry.** If a run fails with a CUDA out-of-memory error,
this halves `--max-image-size` and retries (down to `--floor`, default 800)
— verified end-to-end against a simulated OOM: 2400 → 1200 → 800 succeeded,
with every attempt logged. If it's still OOM at the floor, or fails for a
different reason, it stops and reports rather than looping forever.

```bash
# manual escalation, if the automatic retry isn't enough:
python src/09_dense_stereo.py --scene scene --max-image-size 1600 --cache-size 8
```

This is the slow stage. Expect a real wait even capped at 2400px across 97
images — but this is exactly the workload the 1650's CUDA cores are suited
to, unlike the 56-minute Mac CPU feature-extraction run this project already
did for the sparse stage.

Output: `scene/dense/stereo/`, `output/09_dense_stereo.json`

## Stage 10 — Fuse + mesh

```bash
python src/10_fuse_mesh.py --scene scene
```

Wraps `colmap stereo_fusion` (→ `fused.ply`, a dense point cloud with
per-point colour and normals already sampled from the source photos) then
`colmap poisson_mesher` (→ `meshed-poisson.ply`). COLMAP's Poisson mesher
propagates per-vertex colour from the input cloud, so **this already
produces a coloured mesh** — the safe, always-works texturing baseline, no
extra tool needed.

Prints and logs the numbers the paper's Results section needs — sparse vs.
fused point count (the actual density gain from MVS), mesh vertex/face
count, mesh file size, per-substage timing:

```
--- Dense reconstruction stats (for the paper's Results section) ---
  sparse points (SfM):       64,945
  fused points (MVS):        <N, expect an order of magnitude more>
  densification:             <N>x
  mesh vertices:              <N>
  mesh faces:                 <N>
  mesh has vertex colour:     True
  mesh file size:             <N> MB
```

(Verified end-to-end with a synthetic fused/mesh pair — the point/face
counting and colour detection all read correctly from real PLY headers.)

**Expect a hole or noise above the eaves.** All three elevation bands (low
20 / mid 28 / high 51) are still eye-level-and-up walk-arounds; nothing
looked straight down onto the top of the relief. Report this as a
capture-coverage limitation in the paper, not a bug — the same honest caveat
the plane-proxy sanity check already flagged for the IBR track.

Output: `scene/dense/fused.ply`, `scene/dense/meshed-poisson.ply`,
`output/10_fuse_mesh.json`

## Stage 11 (optional, stretch) — Real UV-texture atlas via OpenMVS

```bash
python src/11_texture_openmvs.py --scene scene --openmvs-bin-dir /path/to/OpenMVS
```

If the Stage 10 vertex-colour mesh looks too coarse for the paper's
figures, OpenMVS's `TextureMesh` gives a proper image-space texture atlas
instead. Wraps `InterfaceCOLMAP` (converts the Stage 8 undistorted workspace
into OpenMVS's project format) → `TextureMesh` (applied to the Stage 10
mesh, so texturing runs on the *same* geometry rather than a different
reconstruction).

**Marked best-effort.** OpenMVS's CLI flag names have changed across
releases and this exact invocation was not executable-tested against a live
OpenMVS install (unlike Stages 7–10, all tested against this repo's real
COLMAP data — see each stage's section above). Confirm flag names with
`TextureMesh --help` on your installed version. See
[troubleshooting.md](troubleshooting.md#stage-11-openmvs) if it doesn't
match.

Output: `scene/openmvs/textured.obj` (+ texture PNG), `output/11_texture_openmvs.json`

## Stage 12 — Export

```bash
python src/12_export_mesh.py scene/dense/meshed-poisson.ply
# or, if Stage 11 ran:
python src/12_export_mesh.py scene/openmvs/textured.obj
```

Uses **trimesh** to convert to `.glb` — a single self-contained,
web-viewable file. Resolves the original plan's open "final export format"
question: `.glb` for viewing/submission, the source `.ply`/`.obj` kept as
the raw/native backup. Handles both a vertex-coloured `.ply` and a
UV-textured `.obj`+`.mtl` generically (verified against both).

Light enough for the Mac — the ROG's job ends at Stage 10 (or 11).

Output: `<mesh>.glb`, `output/12_export_mesh.json`

## Stage 13 — Turntable render

```bash
python src/13_turntable_render.py --mesh scene/dense/meshed-poisson.glb
```

Renders a full 360° orbit to MP4 via Open3D's **legacy
`Visualizer(visible=False)`**, not the newer `OffscreenRenderer` — that was
tried first and fails on macOS with "EGL Headless is not supported on this
platform" (`OffscreenRenderer`'s Filament backend needs EGL, which is
Linux/Windows-only). The legacy Visualizer uses an off-screen GLFW context
instead, verified working headlessly on the Mac, including confirmed
frame-to-frame rotation (not a frozen frame) on a real render.

Because this pipeline recovers actual geometry, the camera path is a full,
arbitrary orbit — unlike the IBR track's virtual-camera arc
(`05_view_dependent_render.py`), which had to stay near the captured ~170°
or sweep off the edge of its single flat plane. That's a genuine upside of
having real 3D structure.

Output: `output/garuda_mesh_turntable.mp4` (default) or `--out`

## Orchestrator

`run_dense_pipeline.py` sequences Stages 7–13 as subprocess calls to the
scripts above, adding `--resume` (skip a stage if its expected output
already exists), `--from`/`--to` (run a subrange), and `--dry-run` (print
every command, run nothing) — all verified: `--resume` correctly skips
already-produced outputs, a failing stage stops the chain and reports
exactly which `--from` to re-run.

```bash
python run_dense_pipeline.py --zip garuda_colmap_data.zip     # full run from a zip
python run_dense_pipeline.py --resume                          # continue after an interruption
python run_dense_pipeline.py --from stereo --to fuse            # just the ROG-side middle
python run_dense_pipeline.py --with-texture                    # include the optional OpenMVS stage
python run_dense_pipeline.py --dry-run                         # see every command first
```

Every stage script also runs stand-alone with its own `--help` — the
orchestrator is a convenience, not a requirement.
