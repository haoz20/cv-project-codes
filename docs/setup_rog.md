# ROG (Windows, GTX 1650) setup

Runs Stage 9 (dense stereo — needs CUDA) and optionally Stage 11 (OpenMVS
texture). Stages 7, 8, 10, 12, 13 can also run here, or on the Mac.

## COLMAP

Download `colmap-x64-windows-cuda.zip` from the
[COLMAP releases page](https://github.com/colmap/colmap/releases) — verified
present as `colmap-x64-windows-cuda.zip` on the current release (`v4.1.1`,
alongside a `colmap-x64-windows-nocuda.zip` — get the CUDA one). Unzip
anywhere. No installer, no build step, no `nvcc`, no Python version pinning
— this is the whole reason the classical route replaced the earlier 3D
Gaussian Splatting plan, which needed a fragile PyTorch/gsplat/Python-3.10
wheel matrix on this exact machine.

**Verify:**

```bat
colmap.exe -h
colmap.exe patch_match_stereo -h
```

- If `colmap.exe -h` fails with a missing-DLL error, install the
  **Visual C++ Redistributable** (x64) — that's the fix, not a sign the
  build is broken.
- `patch_match_stereo -h` should list `--PatchMatchStereo.gpu_index` among
  its options. If it doesn't, you unzipped the `nocuda` build by mistake.

**Make it discoverable by the pipeline scripts** — either:

```bat
set COLMAP_BIN=C:\path\to\COLMAP-3.x-windows-cuda\colmap.exe
```

or add that folder to `PATH`, or pass `--colmap-bin` explicitly to every
stage script (`08_undistort.py`, `09_dense_stereo.py`, `10_fuse_mesh.py`).

## OpenMVS (optional, Stage 11 only)

Only needed if the Stage 10 vertex-colour mesh isn't good enough and you
want a real UV-texture atlas. Download `OpenMVS_Windows_x64_CUDA.7z` from
the [OpenMVS releases page](https://github.com/cdcseacave/openMVS/releases)
— verified present on the current release (`v2.4.0`). Unzip (needs 7-Zip or
similar for `.7z`).

```bat
set OPENMVS_BIN_DIR=C:\path\to\OpenMVS_Windows_x64_CUDA
```

or pass `--openmvs-bin-dir` to `src/11_texture_openmvs.py`. OpenMVS's CLI
flags have changed across releases and this script's exact invocation
wasn't tested against a live install — check `TextureMesh --help` if it
doesn't match; see [troubleshooting.md](troubleshooting.md#stage-11-openmvs).

## Transferring the scene

`garuda_colmap_data.zip` (927 MB, built on the Mac) already bundles the 99
full-resolution JPGs and `sparse/0/` — the one file to move over. Unlike the
abandoned 3DGS plan, **don't pre-downscale the images** before transferring;
COLMAP controls its own working resolution via
`--PatchMatchStereo.max_image_size` (Stage 9), and dense MVS/texture quality
both benefit from the source files staying full resolution.

```bat
python src\07_prepare_scene.py --zip garuda_colmap_data.zip --out scene
```

## Python environment

Only needed on the ROG if running Stages 12/13 (export/render) here too —
otherwise those are lighter run on the Mac. See
[setup_mac.md](setup_mac.md#python-environment) for the same
`pip install -r requirements.txt` step; nothing ROG-specific about it (no
CUDA-matched wheel juggling this time — trimesh/open3d are plain pip
packages, unlike gsplat's Python-3.10-only Windows wheels under the
abandoned plan).

## Quick end-to-end check

```bat
python run_dense_pipeline.py --scene scene --dry-run
```

Prints every command in order without running anything — confirms the
scene path, `colmap.exe`, and (if `--with-texture`) OpenMVS are all
resolved correctly before committing to the multi-hour Stage 9 run.
