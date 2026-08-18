"""
Garuda 3D Reconstruction — shared helpers for Stage 2 (Feature Detection & Matching).

Run all scripts with the Anaconda interpreter: /opt/anaconda3/bin/python
"""

import argparse
import glob
import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def load_image(path, max_dim=1600):
    """
    Load a BGR image and downscale it so its longest side is max_dim.

    Full-resolution phone photos (8064x6048 for our sample set, and likely
    similar for the real Garuda photos) are slow to run SIFT on and make
    keypoint counts hard to compare across images of different original
    sizes. COLMAP downscales internally for the same reason.

    Parameters
    ----------
    path : str
        Path to an image file.
    max_dim : int
        Target size (px) for the longer image dimension.

    Returns
    -------
    img : np.ndarray
        BGR image, resized.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def load_folder(dirpath, max_dim=1600):
    """
    Load all images in a folder, sorted by filename.

    Parameters
    ----------
    dirpath : str
        Folder containing images.
    max_dim : int
        Passed through to load_image.

    Returns
    -------
    list of (name, img) tuples.
    """
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(dirpath, ext)))
    paths = sorted(set(paths))

    if not paths:
        raise FileNotFoundError(f"No images found in {dirpath}")

    return [(os.path.basename(p), load_image(p, max_dim)) for p in paths]


def save_figure(name, show=False):
    """
    Save the current matplotlib figure to output/<name>.png.

    Parameters
    ----------
    name : str
        Output filename (without extension).
    show : bool
        If True, also display the figure interactively (plt.show()).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    if show:
        plt.show()
    plt.close()


def timed(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) and return (result, elapsed_seconds).
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def bgr2rgb(img):
    """Convenience wrapper for matplotlib display (which expects RGB)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_arg_parser(description):
    """
    Shared CLI: --images <folder> to override the input image folder,
    --show to also pop up matplotlib windows instead of only saving PNGs.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--images",
        default=os.path.join(REPO_ROOT, "data", "sample"),
        help="Folder of images to use (default: data/sample)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display figures interactively (in addition to saving them)",
    )
    return parser
