"""
Kratib 3DGS pipeline orchestrator.

Runs the five stage scripts in order, each as its own `python src/0N_*.py`
subprocess. The numbered scripts are the source of truth and stay
individually runnable while tuning -- this just saves retyping the chain
and adds --from / --to / --dry-run ergonomics.

  01 extract   data/kratib.MOV -> frames/*.jpg (+ sharpness.csv)
       >>> manual step: delete blurred frames from frames/ <<<
  02 process   frames/ -> proc/transforms.json      (COLMAP; 80% gate)
  03 train     proc/  -> outputs/<...>/config.yml   (splatfacto, ~30 min)
  04 export    outputs/ -> exports/splat.ply
  05 render    outputs/ -> renders/orbit.mp4

Everything runs on the ROG inside the `kratib` conda env -- see
docs/setup_rog.md.

Usage:
    python run_pipeline.py --dry-run          # print every command, run nothing
    python run_pipeline.py --to process       # stop after COLMAP, then inspect
    python run_pipeline.py --from train       # resume once frames are curated
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO_ROOT, "src")

STAGES = [
    ("extract", "01_extract_frames.py"),
    ("process", "02_process_data.py"),
    ("train", "03_train.py"),
    ("export", "04_export.py"),
    ("render", "05_render.py"),
]
NAMES = [s[0] for s in STAGES]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="from_stage", default=NAMES[0], choices=NAMES)
    parser.add_argument("--to", dest="to_stage", default=NAMES[-1], choices=NAMES)
    parser.add_argument("--fps", type=float, default=2.0, help="Stage 01 sampling rate")
    parser.add_argument("--downscale", type=int, default=2, help="Stage 03 training downscale")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter for the stage scripts (default: this one)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive pause after Stage 01")
    args = parser.parse_args()

    lo, hi = NAMES.index(args.from_stage), NAMES.index(args.to_stage)
    if lo > hi:
        parser.error(f"--from {args.from_stage} comes after --to {args.to_stage}")

    for i, (name, script) in enumerate(STAGES):
        if not (lo <= i <= hi):
            continue

        cmd = [args.python, os.path.join(SRC, script)]
        if name == "extract":
            cmd += ["--fps", str(args.fps)]
        elif name == "train":
            cmd += ["--downscale", str(args.downscale)]
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"\n=== stage: {name} ===")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(f"\nStage '{name}' failed (exit {rc}). Fix, then: "
                     f"python run_pipeline.py --from {name}")

        if name == "extract" and not args.dry_run and i < hi:
            print("\n>>> MANUAL STEP: open frames/sharpness.csv, delete blurred frames")
            print(">>> from frames/, then continue with: python run_pipeline.py --from process")
            if not args.yes:
                resp = input(">>> Continue to COLMAP now? [y/N] ").strip().lower()
                if resp != "y":
                    print("Stopped. Resume with: python run_pipeline.py --from process")
                    return


if __name__ == "__main__":
    main()
