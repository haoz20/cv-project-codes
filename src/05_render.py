"""
Stage 05 -- render an orbit fly-through video.

Wraps `ns-render interpolate`, which flies the camera smoothly through the
training views -- deterministic, no viewer interaction needed. Writes
renders/orbit.mp4.

For a nicer path, open the trained run in `ns-viewer --load-config <cfg>`,
author a camera path in the UI, export its JSON, and run
`ns-render camera-path --camera-path-filename <json> --load-config <cfg>`
instead.

Usage:
    python src/05_render.py
    python src/05_render.py --config outputs/kratib/splatfacto/.../config.yml
    python src/05_render.py --interpolation-steps 15 --dry-run
"""

import argparse
import os

from common import RENDERS_DIR, latest_train_config, require_exe, run


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="splatfacto config.yml (default: most recent under outputs/)")
    parser.add_argument("--output", default=os.path.join(RENDERS_DIR, "orbit.mp4"),
                        help="Output video path (default: renders/orbit.mp4)")
    parser.add_argument("--frame-rate", type=int, default=30,
                        help="Output video fps (default: 30)")
    parser.add_argument("--interpolation-steps", type=int, default=10,
                        help="Interpolated frames between each pair of views (default: 10)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.config:
        config = args.config
    elif args.dry_run:
        config = "<outputs/kratib/splatfacto/LATEST/config.yml>"
    else:
        config = latest_train_config()

    if not args.dry_run:
        require_exe("ns-render", "pip install nerfstudio  (inside the kratib conda env)")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cmd = ["ns-render", "interpolate",
           "--load-config", config,
           "--output-path", args.output,
           "--output-format", "video",
           "--frame-rate", str(args.frame_rate),
           "--interpolation-steps", str(args.interpolation_steps)]
    run(cmd, dry_run=args.dry_run)

    if not args.dry_run and os.path.isfile(args.output):
        mb = os.path.getsize(args.output) / 1e6
        print(f"\n[ok] {args.output}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
