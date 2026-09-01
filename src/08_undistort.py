"""
Stage 8 -- Undistort images for dense reconstruction.

Thin wrapper around `colmap image_undistorter`. Converts the SIMPLE_RADIAL
model (k1=0.031786, see colmap_work/sparse/0/cameras.txt) into undistorted
pinhole images + a matching re-formatted sparse model, which is the input
format colmap's dense stereo stage requires.

CPU/disk-bound, not GPU/VRAM-bound -- can run on the Mac or the ROG.

Input:  <scene>/images/, <scene>/sparse/0/   (see 07_prepare_scene.py)
Output: <scene>/dense/images/, <scene>/dense/sparse/, output/08_undistort.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_common import DEFAULT_SCENE, find_binary, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=DEFAULT_SCENE, help=f"Scene directory (default: {DEFAULT_SCENE})")
    parser.add_argument("--colmap-bin", default=None, help="Path to colmap executable (default: search PATH / COLMAP_BIN)")
    parser.add_argument("--output-type", default="COLMAP", choices=["COLMAP", "PMVS", "CMP-MVS"],
                         help="Dense workspace format (default: COLMAP)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args = parser.parse_args()

    colmap_bin = args.colmap_bin or find_binary("colmap", "COLMAP_BIN")

    image_path = os.path.join(args.scene, "images")
    input_path = os.path.join(args.scene, "sparse", "0")
    output_path = os.path.join(args.scene, "dense")

    if not args.dry_run:
        if not os.path.isdir(image_path):
            parser.error(f"{image_path} not found -- run 07_prepare_scene.py first")
        if not os.path.isdir(input_path):
            parser.error(f"{input_path} not found -- run 07_prepare_scene.py first")
        os.makedirs(output_path, exist_ok=True)

    cmd = [
        colmap_bin, "image_undistorter",
        "--image_path", image_path,
        "--input_path", input_path,
        "--output_path", output_path,
        "--output_type", args.output_type,
    ]

    run_stage("08_undistort", cmd, dry_run=args.dry_run,
               extra_meta={"scene": args.scene, "output_path": output_path})

    if not args.dry_run:
        n_out = len([f for f in os.listdir(os.path.join(output_path, "images"))
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.isdir(os.path.join(output_path, "images")) else 0
        print(f"\nUndistorted {n_out} images -> {output_path}/images")
        print("Next: python src/09_dense_stereo.py --scene", args.scene)


if __name__ == "__main__":
    main()
