"""
Stage 9 -- Dense stereo matching (the one GPU/VRAM-relevant stage).

Wraps `colmap patch_match_stereo`. Unlike 3D Gaussian Splatting training,
dense stereo processes one reference image against its neighbours at a
time -- it never holds all 97 cameras and a giant optimizable parameter
tensor in VRAM simultaneously -- so the working set per step is inherently
much smaller than the (abandoned) splatting plan's.

The one lever that matters at this photo size is
--PatchMatchStereo.max_image_size, which caps the resolution *used
internally* for stereo matching (independent of the on-disk image
resolution -- see 07_prepare_scene.py, which deliberately does NOT
downscale files). At native 8064x6048 this can get tight on a 4 GB card;
2400 is a safe starting point per COLMAP's own guidance for smaller-VRAM
GPUs.

Auto-retry: if a run fails with a CUDA out-of-memory error, this halves
--PatchMatchStereo.max_image_size and retries (down to --floor), rather than
requiring you to notice the failure and manually restart with a smaller
number -- this is the concrete implementation of the plan's Stage 4
escalation ladder. If it OOMs again below the floor, or fails for a
non-memory reason, it stops and reports -- see docs/troubleshooting.md.

Input:  <scene>/dense/  (from 08_undistort.py)
Output: <scene>/dense/stereo/, output/09_dense_stereo.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_common import DEFAULT_SCENE, find_binary, is_oom_error, run_subprocess, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=DEFAULT_SCENE, help=f"Scene directory (default: {DEFAULT_SCENE})")
    parser.add_argument("--colmap-bin", default=None, help="Path to colmap executable (default: search PATH / COLMAP_BIN)")
    parser.add_argument("--max-image-size", type=int, default=2400,
                         help="Starting --PatchMatchStereo.max_image_size (default: 2400; COLMAP's default is 2000)")
    parser.add_argument("--floor", type=int, default=800,
                         help="Don't retry below this max-image-size (default: 800)")
    parser.add_argument("--max-retries", type=int, default=3,
                         help="Max OOM-triggered retries before giving up (default: 3)")
    parser.add_argument("--cache-size", type=int, default=None,
                         help="Optional --PatchMatchStereo.cache_size (GB) to shrink COLMAP's host-side "
                              "image cache if OOM persists at the floor resolution")
    parser.add_argument("--gpu-index", default="0", help="--PatchMatchStereo.gpu_index (default: 0)")
    parser.add_argument("--geom-consistency", default="true", choices=["true", "false"],
                         help="--PatchMatchStereo.geom_consistency (default: true, higher quality, slower)")
    parser.add_argument("--dry-run", action="store_true", help="Print each command without running it")
    args = parser.parse_args()

    colmap_bin = args.colmap_bin or find_binary("colmap", "COLMAP_BIN")
    workspace_path = os.path.join(args.scene, "dense")

    if not args.dry_run and not os.path.isdir(workspace_path):
        parser.error(f"{workspace_path} not found -- run 08_undistort.py first")

    def build_cmd(max_image_size):
        cmd = [
            colmap_bin, "patch_match_stereo",
            "--workspace_path", workspace_path,
            "--PatchMatchStereo.max_image_size", str(max_image_size),
            "--PatchMatchStereo.gpu_index", args.gpu_index,
            "--PatchMatchStereo.geom_consistency", args.geom_consistency,
        ]
        if args.cache_size is not None:
            cmd += ["--PatchMatchStereo.cache_size", str(args.cache_size)]
        return cmd

    current_size = args.max_image_size
    attempts = []
    attempt = 0

    while True:
        print(f"\n--- attempt {attempt + 1}: --PatchMatchStereo.max_image_size {current_size} ---")
        cmd = build_cmd(current_size)
        returncode, stdout, stderr, elapsed = run_subprocess(cmd, dry_run=args.dry_run)
        attempts.append({
            "attempt": attempt + 1,
            "max_image_size": current_size,
            "returncode": returncode,
            "elapsed_seconds": round(elapsed, 2),
            "oom_detected": is_oom_error(stderr) or is_oom_error(stdout),
        })

        if args.dry_run or returncode == 0:
            break

        oom = is_oom_error(stderr) or is_oom_error(stdout)
        if oom and current_size > args.floor and attempt < args.max_retries:
            current_size = max(args.floor, current_size // 2)
            attempt += 1
            print(f"Detected a CUDA out-of-memory failure. Retrying at "
                  f"--PatchMatchStereo.max_image_size {current_size}...")
            continue

        log = {"scene": args.scene, "final_max_image_size": current_size,
               "attempts": attempts, "succeeded": False}
        write_json("09_dense_stereo", log)
        reason = "still OOM at the floor resolution" if oom else "a non-memory-related failure"
        raise RuntimeError(
            f"patch_match_stereo failed after {attempt + 1} attempt(s) ({reason}).\n"
            f"stderr tail:\n{stderr[-1000:]}\n"
            f"See docs/troubleshooting.md. If still OOM at --floor, try --cache-size 8 "
            f"to shrink the host-side cache, or lower --floor further."
        )

    log = {"scene": args.scene, "final_max_image_size": current_size,
           "attempts": attempts, "succeeded": True}
    write_json("09_dense_stereo", log)

    if not args.dry_run:
        print(f"\nDense stereo done at max_image_size={current_size} "
              f"({'no retries needed' if attempt == 0 else f'{attempt} retry/retries'})")
        print("Next: python src/10_fuse_mesh.py --scene", args.scene)


if __name__ == "__main__":
    main()
