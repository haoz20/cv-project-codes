"""
Stage 02 -- frames -> COLMAP poses (nerfstudio-native).

Thin wrapper over `ns-process-data images`, which runs COLMAP (feature
extraction, matching, mapping) and writes proc/transforms.json in
nerfstudio's own convention.

We deliberately do NOT use instant-ngp's colmap2nerf.py: it emits
aabb_scale + OpenGL-convention poses that nerfstudio then has to
reinterpret, a known source of silent scale/pose bugs. Running COLMAP as
its own stage here also lets the blurred-frame deletion happen *between*
extraction and COLMAP.

GO / NO-GO GATE: this script prints the registered-image count at the end.
If fewer than ~80% of the frames registered, stop -- the pose graph is bad
and no splatfacto tuning fixes it. Reshoot per docs/capture.md (more
overlap, keep the textured background in frame).

By default we pass --skip-image-processing and --num-downscales 0:
ns-process-data's image copy/downscale step builds ffmpeg commands with
bash-style quoting that breaks on Windows (`-map ''` -> empty), and the
Stage 01 frames are already clean JPEGs that don't need re-encoding.
COLMAP then reads the frames folder directly and transforms.json
references them by relative path. Pass --copy-images to restore the
normal copy/downscale behaviour (fine on Linux).

Usage:
    python src/02_process_data.py                 # frames/ -> proc/
    python src/02_process_data.py --data frames --output proc
    python src/02_process_data.py --no-gpu        # CPU SIFT (WSL2 without a GL context)
    python src/02_process_data.py --copy-images --num-downscales 3
    python src/02_process_data.py --dry-run
"""

import argparse
import json
import os
import sys

from common import FRAMES_DIR, PROC_DIR, count_images, require_exe, run


def summarize(proc_dir, n_src):
    tj = os.path.join(proc_dir, "transforms.json")
    if not os.path.isfile(tj):
        sys.exit(f"ERROR: {tj} not written -- ns-process-data did not finish.")
    with open(tj) as f:
        data = json.load(f)
    n_registered = len(data.get("frames", []))

    print("\n=== Stage 02 result ===")
    print(f"  registered frames : {n_registered}")
    if n_src:
        pct = 100.0 * n_registered / n_src
        print(f"  source frames     : {n_src}")
        print(f"  registration rate : {pct:.0f}%")
        if pct < 80:
            print("\n  *** BELOW 80% -- do not train on this. Reshoot per docs/capture.md. ***")
        else:
            print("\n  OK -- proceed to  python src/03_train.py")
    print(f"  transforms.json   : {tj}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=FRAMES_DIR,
                        help="Folder of frames (default: frames/)")
    parser.add_argument("--output", default=PROC_DIR,
                        help="Output dir (default: proc/)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Run COLMAP feature extraction/matching on CPU. Needed in "
                             "some WSL2 setups where SiftGPU has no OpenGL context.")
    parser.add_argument("--num-downscales", type=int, default=0,
                        help="Pre-generated downscaled image sets (default: 0 -- "
                             "splatfacto downscales in Python).")
    parser.add_argument("--copy-images", action="store_true",
                        help="Let ns-process-data copy/normalise images via ffmpeg. "
                             "Default is to skip it (--skip-image-processing) -- that "
                             "ffmpeg step is broken on Windows and the frames are "
                             "already clean JPEGs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n_src = count_images(args.data)

    if not args.dry_run:
        if not os.path.isdir(args.data):
            sys.exit(f"ERROR: frames folder not found: {args.data}\n"
                     f"       Run src/01_extract_frames.py first.")
        if n_src == 0:
            sys.exit(f"ERROR: no images in {args.data}")
        require_exe("ns-process-data",
                    "pip install nerfstudio  (inside the kratib conda env)")
        require_exe("colmap",
                    "Install COLMAP on PATH -- see docs/setup_wsl2.md (recommended) "
                    "or docs/setup_rog.md")

    cmd = ["ns-process-data", "images",
           "--data", args.data,
           "--output-dir", args.output,
           "--num-downscales", str(args.num_downscales)]
    if not args.copy_images:
        cmd.append("--skip-image-processing")
    if args.no_gpu:
        cmd.append("--no-gpu")
    run(cmd, dry_run=args.dry_run)

    if not args.dry_run:
        summarize(args.output, n_src)


if __name__ == "__main__":
    main()
