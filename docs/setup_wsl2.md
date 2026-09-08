# ROG setup via WSL2 (recommended)

Native-Windows setup for this pipeline hits one toolchain wall after another
(Smart App Control blocking unsigned DLLs, corrupt-DLL "Bad Image" errors,
COLMAP option-name churn, and finally `nvcc`'s `cudafe++` crashing because
CUDA 12.4 doesn't support the current MSVC). WSL2 sidesteps all of it:

- `pip install gsplat` pulls a **prebuilt Linux wheel** -- no CUDA compile,
  no MSVC, no `cudafe++`.
- `apt install colmap ffmpeg` -- signed packages, no Smart App Control.
- The RTX GPU passes through via the **Windows** NVIDIA driver.
- The pipeline scripts (`src/*.py`, `run_pipeline.py`) run **unchanged**.

Setup is ~30 min. Do everything below inside the Ubuntu terminal unless it
says "(Windows)".

---

## 1. Install WSL2 + Ubuntu  (Windows, PowerShell as Administrator)

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot if asked. On first launch Ubuntu prompts for a username + password
(local to WSL -- pick anything, remember the password for `sudo`).

> Already have WSL with a different distro? `wsl --install -d Ubuntu-22.04`
> still adds this one. Ubuntu 22.04 is the safe choice; 24.04 also works.

**Do NOT install an NVIDIA driver inside Ubuntu.** WSL uses the Windows
driver. Just verify the GPU is visible:

```bash
nvidia-smi
```

Should list the RTX card. If it doesn't, update the GeForce driver on
Windows (GeForce Experience or nvidia.com), then reopen Ubuntu.

## 2. System packages

```bash
sudo apt update
sudo apt install -y build-essential git curl wget colmap ffmpeg
```

`colmap` from apt is CUDA-enabled and old enough to use the option names
nerfstudio expects (unlike the 3.12 Windows build that broke earlier).

## 3. Miniconda (inside Ubuntu)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
exec bash
```

## 4. Environment

No `environment.yml` here -- WSL doesn't need the in-env CUDA toolkit
(that was only for gsplat's Windows compile). Plain pip:

```bash
conda create -n kratib python=3.11 -y
conda activate kratib
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install nerfstudio av opencv-python numpy
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt26cu124
```

The gsplat line installs a **prebuilt** Linux wheel (torch 2.6 + CUDA
12.4). Verify -- this should print with no compile:

```bash
python -c "from gsplat.cuda._backend import _C; print('gsplat OK')"
```

## 5. Repo + video

Keep everything on the **Linux filesystem** (`~/...`), never under
`/mnt/c/...` -- cross-boundary file I/O is slow and COLMAP/training are
I/O-heavy.

```bash
cd ~
git clone https://github.com/haoz20/cv-project-codes.git
cd cv-project-codes
```

Copy the capture video in from Windows (adjust the source path):

```bash
cp /mnt/c/Users/<you>/Documents/znhao-project/cv-project-codes/data/kratib_up.mp4 data/
```

(`kratib_up.mp4` = the ffmpeg-uprighted file from the rotation fix. If you
only have the original `.MOV`, copy that and re-do the upright step from
`docs/capture.md` here.)

## 6. Run

```bash
python src/01_extract_frames.py --video data/kratib_up.mp4
#   ... open frames/sharpness.csv, delete blurred frames ...
python src/02_process_data.py
#   -> if COLMAP fails with an OpenGL / SiftGPU context error, use:
#      python src/02_process_data.py --no-gpu
#   STOP if registration < 80%
python src/03_train.py
python src/04_export.py
python src/05_render.py
```

## 7. Viewer

`ns-train` serves the viewer on `0.0.0.0:7007`. WSL2 forwards localhost, so
just open **`http://localhost:7007`** in your normal Windows browser while
training runs.

## Notes

- First `ns-train` in WSL still "sets up CUDA" for a few seconds but does
  **not** compile -- the prebuilt wheel already has the kernels.
- `nvidia-smi` inside WSL shows a slightly lower driver/CUDA version string
  than Windows; that's expected and fine.
- 6 GB VRAM still applies: the repo already trains with `--sh-degree 2` and
  `--downscale-factor 2`. If it OOMs, `python src/03_train.py --downscale 3`.
