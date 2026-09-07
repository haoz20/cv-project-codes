"""
Stage 03 -- train splatfacto (3D Gaussian Splatting).

Wraps `ns-train splatfacto` with the settings the plan fixed for a 6 GB
card:

  --pipeline.model.sh-degree 2         fewer SH coeffs per Gaussian -> less VRAM
  nerfstudio-data --downscale-factor 2 train at ~1080p, not 4K; also keeps
                                       splatfacto's in-RAM image cache small

Frames stay full-resolution on disk (COLMAP wanted the detail) -- the
downscale is a training-time argument only.

Get ONE complete run at these settings before touching any quality knob. A
finished mediocre splat beats an OOM at iteration 12,000. If 6 GB still
OOMs during densification, the levers in order are: --downscale 3, then an
MCMC strategy with a Gaussian cap (see nerfstudio splatfacto docs).

The live viewer is at http://localhost:7007 while training.

Run the FIRST training from an "x64 Native Tools Command Prompt for VS
2022" with the kratib env active -- gsplat JIT-compiles its CUDA kernels on
first import and needs nvcc + cl.exe on PATH. See docs/setup_rog.md.

Usage:
    python src/03_train.py                       # proc/ -> outputs/
    python src/03_train.py --downscale 3         # if 6 GB still OOMs
    python src/03_train.py --max-iters 30000
    python src/03_train.py --dry-run
"""

import argparse
import os
import sys

from common import OUTPUTS_DIR, PROC_DIR, require_exe, run


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=PROC_DIR,
                        help="ns-process-data output (default: proc/)")
    parser.add_argument("--output", default=OUTPUTS_DIR,
                        help="Training output root (default: outputs/)")
    parser.add_argument("--experiment-name", default="kratib",
                        help="nerfstudio experiment name (default: kratib)")
    parser.add_argument("--sh-degree", type=int, default=2,
                        help="Spherical-harmonics degree (default: 2)")
    parser.add_argument("--downscale", type=int, default=2,
                        help="Training-time downscale factor (default: 2)")
    parser.add_argument("--max-iters", type=int, default=30000,
                        help="Iterations (default: 30000)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        tj = os.path.join(args.data, "transforms.json")
        if not os.path.isfile(tj):
            sys.exit(f"ERROR: {tj} not found. Run src/02_process_data.py first.")
        require_exe("ns-train", "pip install nerfstudio  (inside the kratib conda env)")

    cmd = ["ns-train", "splatfacto",
           "--data", args.data,
           "--output-dir", args.output,
           "--experiment-name", args.experiment_name,
           "--pipeline.model.sh-degree", str(args.sh_degree),
           "--max-num-iterations", str(args.max_iters),
           "--viewer.quit-on-train-completion", "True",
           "nerfstudio-data",
           "--downscale-factor", str(args.downscale)]
    run(cmd, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nNext: python src/04_export.py   (writes exports/splat.ply)")


if __name__ == "__main__":
    main()
