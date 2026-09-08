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

## Get the capture video

The `kratib.MOV` capture is not in git (large binary). Download it from
Google Drive and place it at `data/kratib.MOV`:

<https://drive.google.com/drive/folders/1A1RrD60YqDWfjYWO6MZWkZ678ztJa5qo?usp=sharing>

## Setup (on the ROG)

**Use WSL2.** Native-Windows setup fails at the gsplat CUDA build (CUDA 12.4
vs current MSVC -- `cudafe++` crashes), on top of Smart App Control, DLL,
and COLMAP-version issues. In WSL2, `gsplat` installs as a prebuilt wheel
and `colmap` is one `apt` line. Full steps:

**[`docs/setup_wsl2.md`](docs/setup_wsl2.md)** ← start here

Then:

```bash
python src/01_extract_frames.py --video data/kratib_up.mp4
#   ... delete blurred frames from frames/ ...
python src/02_process_data.py            # add --no-gpu if COLMAP hits a GL error
python src/03_train.py                   # viewer at http://localhost:7007
python src/04_export.py
python src/05_render.py
```

`environment.yml` / `requirements.txt` / [`docs/setup_rog.md`](docs/setup_rog.md)
are the native-Windows attempt, kept for reference.
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
