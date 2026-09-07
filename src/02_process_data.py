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

Usage:
    python src/02_process_data.py                 # frames/ -> proc/
    python src/02_process_data.py --data frames --output proc
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
                    "Install COLMAP (Windows CUDA build) on PATH -- see docs/setup_rog.md")

    cmd = ["ns-process-data", "images",
           "--data", args.data,
           "--output-dir", args.output]
    run(cmd, dry_run=args.dry_run)

    if not args.dry_run:
        summarize(args.output, n_src)


if __name__ == "__main__":
    main()
