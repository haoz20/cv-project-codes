"""
Kratib 3DGS pipeline -- shared helpers.

Every stage script (src/0N_*.py) imports from here: repo-relative paths, a
subprocess runner with --dry-run / timing / command echo, a PATH-executable
check that fails loudly with a fix hint instead of a bare FileNotFoundError,
and a locator for the most recent splatfacto training config.

Run everything inside the `kratib` conda env -- see docs/setup_rog.md.
"""

import glob
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(REPO_ROOT, "data")
FRAMES_DIR = os.path.join(REPO_ROOT, "frames")
PROC_DIR = os.path.join(REPO_ROOT, "proc")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")   # ns-train writes here
EXPORTS_DIR = os.path.join(REPO_ROOT, "exports")
RENDERS_DIR = os.path.join(REPO_ROOT, "renders")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
VIDEO_EXTS = (".mov", ".MOV", ".mp4", ".MP4", ".m4v", ".mkv", ".avi")


def find_capture_video(data_dir=DATA_DIR, stem="kratib"):
    """
    Locate the capture video. Prefers data/<stem>.<ext> for a known video
    extension (any of VIDEO_EXTS -- .MOV and .mp4 both fine); otherwise, if
    data/ holds exactly one video file, use that. Returns a path or None.
    """
    for ext in VIDEO_EXTS:
        p = os.path.join(data_dir, stem + ext)
        if os.path.isfile(p):
            return p
    if os.path.isdir(data_dir):
        vids = [f for f in os.listdir(data_dir) if f.endswith(VIDEO_EXTS)]
        if len(vids) == 1:
            return os.path.join(data_dir, vids[0])
    return None


def require_exe(name, hint):
    """Return the resolved path to `name`, or exit with a fix hint."""
    path = shutil.which(name)
    if path is None:
        sys.exit(f"ERROR: '{name}' not found on PATH.\n       {hint}")
    return path


def run(cmd, dry_run=False, cwd=None):
    """Echo and run a command list. Return elapsed seconds. Exit on failure."""
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}")
    if dry_run:
        return 0.0
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        sys.exit(f"\nCommand failed (exit {proc.returncode}): {printable}")
    print(f"[done] {elapsed:.1f}s")
    return elapsed


def count_images(dirpath):
    """Number of image files directly in dirpath (0 if it does not exist)."""
    if not os.path.isdir(dirpath):
        return 0
    return sum(1 for f in os.listdir(dirpath) if f.endswith(IMAGE_EXTS))


def latest_train_config(outputs_dir=OUTPUTS_DIR):
    """
    Path to the most recent splatfacto config.yml.

    nerfstudio writes outputs/<experiment>/splatfacto/<timestamp>/config.yml.
    """
    hits = sorted(glob.glob(os.path.join(outputs_dir, "*", "splatfacto", "*", "config.yml")))
    if not hits:
        sys.exit(f"ERROR: no splatfacto config.yml under {outputs_dir}.\n"
                 f"       Run src/03_train.py first, or pass --config explicitly.")
    return hits[-1]
