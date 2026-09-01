"""
Garuda 3D Reconstruction -- dense pipeline orchestrator.

Sequences Stages 7-13 (scene prep -> undistort -> dense stereo -> fuse+mesh
-> [optional texture] -> export -> turntable render), each as its own
`python src/0N_*.py` subprocess -- the numbered stage scripts are the source
of truth and can always be run individually while tuning; this just saves
retyping the chain and adds --resume/--from/--to/--dry-run ergonomics.

Two machines are involved, per docs/dense_pipeline.md:
  - Stages 7 (prep), 8 (undistort, CPU-only) can run on either machine.
  - Stages 9 (dense stereo) and 11 (OpenMVS texture, optional) need CUDA --
    that's the ROG (GTX 1650). Stage 10 (fusion + Poisson) also needs COLMAP
    but is comparatively light; runs wherever COLMAP is installed.
  - Stages 12 (export) and 13 (render) are light enough for the Mac.

Usage:
    python run_dense_pipeline.py --zip garuda_colmap_data.zip
    python run_dense_pipeline.py --resume                      # skip stages already done
    python run_dense_pipeline.py --from stereo --to fuse
    python run_dense_pipeline.py --dry-run                     # print every command, run nothing
    python run_dense_pipeline.py --with-texture                # include the optional OpenMVS stage

Each stage is still runnable on its own -- see the "Stages individually"
section of docs/dense_pipeline.md.
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO_ROOT, "src")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

# (short name, script, extra CLI args (list, may reference {scene}/{mesh}/{glb}),
#  expected-output-path template for --resume, included by default)
STAGES = [
    ("prepare", "07_prepare_scene.py", [], "{scene}/images", True),
    ("undistort", "08_undistort.py", [], "{scene}/dense/images", True),
    ("stereo", "09_dense_stereo.py", [], "{scene}/dense/stereo", True),
    ("fuse", "10_fuse_mesh.py", [], "{scene}/dense/meshed-poisson.ply", True),
    ("texture", "11_texture_openmvs.py", [], "{scene}/openmvs/textured.obj", False),
    ("export", "12_export_mesh.py", ["{mesh}"], "{glb}", True),
    ("render", "13_turntable_render.py", ["--mesh", "{glb}"], "{video}", True),
]
STAGE_NAMES = [s[0] for s in STAGES]


def resolve_path(template, ctx):
    return template.format(**ctx)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=os.path.join(REPO_ROOT, "scene"), help="Scene working directory")
    parser.add_argument("--zip", default=None, help="Passed through to 07_prepare_scene.py --zip")
    parser.add_argument("--images", default=None, help="Passed through to 07_prepare_scene.py --images")
    parser.add_argument("--sparse", default=None, help="Passed through to 07_prepare_scene.py --sparse")
    parser.add_argument("--colmap-bin", default=None, help="Passed through to stages 8-11")
    parser.add_argument("--openmvs-bin-dir", default=None, help="Passed through to stage 11")
    parser.add_argument("--video-out", default=os.path.join(OUTPUT_DIR, "garuda_mesh_turntable.mp4"))
    parser.add_argument("--from", dest="from_stage", default=STAGE_NAMES[0], choices=STAGE_NAMES)
    parser.add_argument("--to", dest="to_stage", default=STAGE_NAMES[-1], choices=STAGE_NAMES)
    parser.add_argument("--with-texture", action="store_true",
                         help="Include the optional OpenMVS UV-texture stage (off by default)")
    parser.add_argument("--resume", action="store_true", help="Skip a stage if its expected output already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print every command; run nothing")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to run stage scripts with")
    args = parser.parse_args()

    mesh_source = "{scene}/openmvs/textured.obj" if args.with_texture else "{scene}/dense/meshed-poisson.ply"
    ctx = {
        "scene": args.scene,
        "mesh": None,  # filled in once we know whether texture ran
        "glb": None,
        "video": args.video_out,
    }
    ctx["mesh"] = resolve_path(mesh_source, ctx)
    ctx["glb"] = os.path.splitext(ctx["mesh"])[0] + ".glb"

    from_idx = STAGE_NAMES.index(args.from_stage)
    to_idx = STAGE_NAMES.index(args.to_stage)
    if from_idx > to_idx:
        parser.error(f"--from {args.from_stage} comes after --to {args.to_stage}")

    run_log = {"scene": args.scene, "dry_run": args.dry_run, "stages": []}
    overall_start = time.perf_counter()

    for i, (name, script, extra_args, output_template, default_on) in enumerate(STAGES):
        if not (from_idx <= i <= to_idx):
            continue
        if name == "texture" and not args.with_texture:
            print(f"[skip] {name} (pass --with-texture to include it)")
            continue

        expected_output = resolve_path(output_template, ctx)
        if args.resume and os.path.exists(expected_output):
            print(f"[skip] {name} (--resume: {expected_output} already exists)")
            run_log["stages"].append({"stage": name, "skipped": True})
            continue

        cmd = [args.python, os.path.join(SRC, script)]
        if name == "prepare":
            if args.zip:
                cmd += ["--zip", args.zip]
            elif args.images and args.sparse:
                cmd += ["--images", args.images, "--sparse", args.sparse]
            cmd += ["--out", args.scene]
        elif name in ("undistort", "stereo", "fuse"):
            cmd += ["--scene", args.scene]
            if args.colmap_bin:
                cmd += ["--colmap-bin", args.colmap_bin]
        elif name == "texture":
            cmd += ["--scene", args.scene]
            if args.openmvs_bin_dir:
                cmd += ["--openmvs-bin-dir", args.openmvs_bin_dir]
        cmd += [resolve_path(a, ctx) if "{" in a else a for a in extra_args]
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"\n=== stage: {name} ===")
        stage_start = time.perf_counter()
        if args.dry_run:
            print("$", " ".join(cmd))
            returncode = 0
        else:
            proc = subprocess.run(cmd)
            returncode = proc.returncode
        elapsed = time.perf_counter() - stage_start

        run_log["stages"].append({"stage": name, "command": cmd, "returncode": returncode,
                                   "elapsed_seconds": round(elapsed, 2)})

        if returncode != 0:
            run_log["succeeded"] = False
            run_log["failed_stage"] = name
            _write_run_log(run_log, overall_start)
            print(f"\nStage '{name}' failed (exit {returncode}). Stopping.")
            print(f"Fix and re-run with --resume --from {name} to continue from here.")
            sys.exit(returncode)

    run_log["succeeded"] = True
    _write_run_log(run_log, overall_start)
    if not args.dry_run and to_idx == len(STAGES) - 1:
        print(f"\nDone. Turntable video: {args.video_out}")


def _write_run_log(run_log, overall_start):
    run_log["total_elapsed_seconds"] = round(time.perf_counter() - overall_start, 2)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "dense_pipeline_run.json")
    with open(path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
