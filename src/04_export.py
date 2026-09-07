"""
Stage 04 -- export the trained splat to a portable .ply.

Wraps `ns-export gaussian-splat`, which reads the latest training
checkpoint and writes exports/splat.ply in the standard 3DGS PLY format.
That file opens on the Mac with no install at all -- drag it into
https://supersplat.playcanvas.com in a browser.

A 200-frame scene can produce a 200-500 MB .ply; it is gitignored. Use
SuperSplat to crop floaters and export a compressed .splat / .sog for
sharing or for the report.

Usage:
    python src/04_export.py                  # latest run -> exports/splat.ply
    python src/04_export.py --config outputs/kratib/splatfacto/2026-.../config.yml
    python src/04_export.py --dry-run
"""

import argparse
import os

from common import EXPORTS_DIR, latest_train_config, require_exe, run


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="splatfacto config.yml (default: most recent under outputs/)")
    parser.add_argument("--output", default=EXPORTS_DIR,
                        help="Output dir (default: exports/)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.config:
        config = args.config
    elif args.dry_run:
        config = "<outputs/kratib/splatfacto/LATEST/config.yml>"
    else:
        config = latest_train_config()

    if not args.dry_run:
        require_exe("ns-export", "pip install nerfstudio  (inside the kratib conda env)")

    cmd = ["ns-export", "gaussian-splat",
           "--load-config", config,
           "--output-dir", args.output]
    run(cmd, dry_run=args.dry_run)

    if not args.dry_run:
        ply = os.path.join(args.output, "splat.ply")
        if os.path.isfile(ply):
            mb = os.path.getsize(ply) / 1e6
            print(f"\n[ok] {ply}  ({mb:.0f} MB)")
            print("     View on the Mac: drag into https://supersplat.playcanvas.com")
        print("\nNext: python src/05_render.py   (writes renders/orbit.mp4)")


if __name__ == "__main__":
    main()
