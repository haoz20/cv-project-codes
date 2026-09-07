# ROG setup (Windows, RTX 40-series 6 GB)

Everything in this pipeline runs here, inside a conda env named `kratib`.
The Mac is only for writing and for viewing the exported `.ply` in a
browser.

> **Why conda, not uv/venv:** on Windows, conda installs the CUDA toolkit
> *into the environment*, so `nvcc` (which gsplat's build needs) arrives
> without a system-wide CUDA install or PATH juggling. It is also
> nerfstudio's documented Windows path, so error messages match their docs.
> An earlier attempt at this project on a "fragile PyTorch/gsplat/Python
> wheel matrix" was abandoned -- the ordered procedure below is the fix.

## 1. Manual prerequisites (not Python packages -- install once)

| Component | Notes |
|---|---|
| **NVIDIA driver** | A recent GeForce driver. `nvidia-smi` must work. |
| **VS 2022 Build Tools** | Installer -> check **"Desktop development with C++"**. Provides `cl.exe`, which gsplat's CUDA build invokes. |
| **COLMAP** | Download `COLMAP-*-windows-cuda.zip` from the [COLMAP releases page](https://github.com/colmap/colmap/releases). Unzip anywhere, add that folder to `PATH`. Verify: `colmap -h` runs, and `colmap patch_match_stereo -h` lists `--PatchMatchStereo.gpu_index` (if not, you unzipped the `nocuda` build). If `colmap -h` fails on a missing DLL, install the x64 **Visual C++ Redistributable**. |

## 2. Create the environment

```bat
conda env create -f environment.yml
conda activate kratib
```

This gives you Python 3.11, an in-env CUDA 12.4 toolkit (`nvcc`), and
`ffmpeg`.

## 3. Install the pip half -- order matters

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install gsplat --no-build-isolation
```

- `torch` comes from the **cu124 index**, not PyPI -- the default PyPI
  wheel is CPU-only on Windows.
- `gsplat` is **separate and last**: its build script imports `torch`, so
  torch must already be installed, and `--no-build-isolation` is required
  so the build can see it.

## 4. First-run gotcha: the CUDA compile

`gsplat` **JIT-compiles its CUDA kernels on first import**, and needs both
`nvcc` and `cl.exe` on `PATH` at that moment. Run the **first** training
from an **"x64 Native Tools Command Prompt for VS 2022"** (Start menu),
then `conda activate kratib` inside it. After the kernels are built once,
an ordinary terminal is fine.

## 5. Verify

```bat
python -c "import torch; print(torch.cuda.is_available())"     REM -> True
nvcc --version                                                 REM -> 12.4, from the env
python -c "import gsplat"                                       REM compiles once, then clean
python run_pipeline.py --dry-run                               REM prints every command
```

## 6. Run

```bat
python run_pipeline.py            REM full chain, pauses after frame extraction
```

or stage by stage:

```bat
python src\01_extract_frames.py
REM  ... delete blurred frames from frames\ using frames\sharpness.csv ...
python src\02_process_data.py    REM STOP if registration < 80%
python src\03_train.py           REM ~30 min; live viewer at http://localhost:7007
python src\04_export.py          REM -> exports\splat.ply
python src\05_render.py          REM -> renders\orbit.mp4
```

## If the 6 GB card OOMs during training

In order:

1. `python src\03_train.py --downscale 3`
2. Switch to an MCMC densification strategy with a hard Gaussian cap
   (~500k) -- see the [splatfacto docs](https://docs.nerf.studio/nerfology/methods/splat.html).
3. Fallback: train on a cloud GPU from the same `proc/` folder; nothing
   upstream of Stage 03 changes.
