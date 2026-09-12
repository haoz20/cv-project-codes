# kratib-3dgs

Neural rendering of a **kratib** (woven Thai sticky-rice basket -- Thai art
and culture) with **3D Gaussian Splatting**, via nerfstudio's `splatfacto`.

CV term project, Path 2 (Neural Rendering). One handheld video in; a
`splat.ply` you can open in a browser and an orbit fly-through `.mp4` out.

> Successor to the Garuda classical dense-MVS project
> (`../cv-project-codes/`, GitHub `haoz20/cv-project-codes`), which stays as
> the archive.

## Hardware

| Machine | Role |
|---|---|
| **ASUS ROG**, RTX 40-series 6 GB | Everything: frame extraction, COLMAP, training, export, render |
| **MacBook Air M2** | Writing, and viewing `exports/splat.ply` in a browser (no install) |

## Pipeline

| Stage | Script | In -> Out |
|---|---|---|
| 01 extract | `src/01_extract_frames.py` | `data/kratib.MOV` (or `.mp4`) -> `frames/*.jpg` + `sharpness.csv` |
| -- manual -- | -- | delete blurred frames from `frames/` (worst-first order in `sharpness.csv`) |
| 02 process | `src/02_process_data.py` | `frames/` -> `proc/transforms.json` (COLMAP). **Stop if registration < 80%.** |
| 03 train | `src/03_train.py` | `proc/` -> `outputs/kratib/splatfacto/<ts>/` (~30 min; viewer at `localhost:7007`) |
| 04 export | `src/04_export.py` | `outputs/` -> `exports/splat.ply` |
| 05 render | `src/05_render.py` | `outputs/` -> `renders/orbit.mp4` |

The numbered scripts are the source of truth and run individually.
`run_pipeline.py` just chains them with `--from` / `--to` / `--dry-run`.

To export a specific training run (rather than the latest one under
`outputs/`) to its own named folder -- useful for keeping multiple
experiments' `.ply` files side by side instead of overwriting
`exports/splat.ply` each time -- call `ns-export` directly:

```bat
ns-export gaussian-splat --load-config outputs\kratib\splatfacto\<ts>\config.yml --output-dir exports\splat\
```

## Get the capture video

The `kratib.MOV` capture is not in git (large binary). Download it from
Google Drive and place it at `data/kratib.MOV`:

<https://drive.google.com/drive/folders/1A1RrD60YqDWfjYWO6MZWkZ678ztJa5qo?usp=sharing>

## Setup (on the ROG, native Windows)

Building `gsplat` from source on this machine fails (`cudafe++` crashes --
CUDA vs the current MSVC). The way through is a **prebuilt gsplat wheel**:
no CUDA toolkit, no MSVC, no compile. The wheel pins the Python and torch
versions:

- **Python 3.10** -- the wheel is `cp310`
- **torch 2.4.1 + cu118** -- the wheel is `+pt24cu118`; torch bundles the
  CUDA 11.8 runtime, so nothing else CUDA is needed
  (torch 2.1.2 will *not* load a `+pt24` wheel -- ABI mismatch)

```bat
conda create -n kratib python=3.10 -y
conda activate kratib
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install gsplat
python -c "import gsplat; from gsplat.cuda._backend import _C; print('gsplat', gsplat.__version__, 'OK')"
```

The last line must print `... OK` **with no compilation**. (The second
gsplat install re-pins the wheel in case `pip install nerfstudio` pulled a
different gsplat; splatfacto works with 1.5.x.)

**COLMAP:** download `colmap-x64-windows-cuda.zip` from a **3.11.x**
release (not 3.12 -- its option names changed and break `ns-process-data`),
unzip, add its folder to `PATH`. See [`docs/setup_rog.md`](docs/setup_rog.md).

Then run the pipeline:

```bat
python src\01_extract_frames.py --video data\kratib_up.mp4
REM  ... delete blurred frames from frames\ using frames\sharpness.csv ...
python src\02_process_data.py
python src\03_train.py
python src\04_export.py
python src\05_render.py
```

Fallback if native Windows still fights you:
[`docs/setup_wsl2.md`](docs/setup_wsl2.md).
How to shoot the video: [`docs/capture.md`](docs/capture.md).

## Viewing the result on the Mac

Copy `exports/splat.ply` over, open <https://supersplat.playcanvas.com>,
drag the file in. No nerfstudio, torch, or CUDA needed on the Mac.

## Scope

In: the working pipeline + the fly-through video. Out (for now):
PSNR/SSIM/LPIPS eval, ablations, NeRF baseline. `ns-eval` is a one-liner if
that changes.

## Settings fixed for the 6 GB card

`--pipeline.model.sh-degree 2` and training-time `--downscale-factor 2`
(frames stay full-res on disk). If it still OOMs: `--downscale 3`, then an
MCMC strategy with a Gaussian cap. See `docs/setup_rog.md`.
