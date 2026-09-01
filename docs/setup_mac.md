# Mac setup

Runs Stages 7 (prep), 8 (undistort, CPU-only), 12 (export), 13 (render) —
and 10 (fuse+mesh) if COLMAP is installed here too (`brew install colmap`
already gives you a working, if CPU-only for dense stereo, COLMAP — that's
enough for Stages 8 and 10, just not Stage 9). Stage 9 (dense stereo) and
Stage 11 (OpenMVS texture) need CUDA and run on the ROG.

## Python environment

This repo's existing scripts (`01`–`06`) already run under
`/opt/anaconda3/bin/python`, with cv2 4.12.0 / numpy 1.26.4 / matplotlib
3.9.2. The dense-pipeline scripts (`07`–`13`) add `trimesh` and `open3d`:

```bash
/opt/anaconda3/bin/pip install -r requirements.txt
```

Both installed and were exercised end-to-end during development of this
pipeline: `trimesh` (5.1.0 at time of writing) correctly loads a
vertex-coloured `.ply` and exports a valid `.glb` (confirmed via `file` —
"glTF binary model, version 2"); `open3d` (0.19.0) renders a real,
non-frozen 360° orbit (confirmed via frame-to-frame pixel differencing on a
test render).

## A real gotcha this pipeline already hit

**`open3d.visualization.rendering.OffscreenRenderer` does not work
headlessly on macOS** — it fails with:

```
[Open3D Error] EGL Headless is not supported on this platform.
```

`OffscreenRenderer`'s Filament backend needs EGL, which is Linux/Windows
only. `13_turntable_render.py` uses the **legacy**
`o3d.visualization.Visualizer(visible=False)` instead, which uses an
off-screen GLFW context and was verified working on this machine. If you
ever rewrite that stage, don't switch back to `OffscreenRenderer` on a Mac.

## Which stage produces what to move to the ROG

`07_prepare_scene.py` can also run here first, to sanity-check the scene
bundle before the 927 MB transfer:

```bash
python src/07_prepare_scene.py --images images --sparse colmap_work/sparse/0 --out /tmp/scene_check
```

This is exactly how the script was validated during development — against
the real `images/` (99 files, split across `high/low/mid` subfolders — the
script flattens this automatically since the sparse model's `images.txt`
uses a flat namespace) and `colmap_work/sparse/0/` (97 registered poses,
64,945 points) already in this repo. It reported those exact numbers with
no warnings.
