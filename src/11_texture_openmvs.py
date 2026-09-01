"""
Stage 11 (stretch, optional) -- Real UV-texture atlas via OpenMVS.

10_fuse_mesh.py already produces a coloured mesh (per-vertex colour, no
extra tool). This stage upgrades that to a proper image-space texture atlas
if the vertex-colour result looks too coarse for the paper's figures.

Also a verified prebuilt Windows CUDA binary (OpenMVS_Windows_x64_CUDA.7z,
confirmed present on OpenMVS's current GitHub release) -- no compilation
here either.

This stage is marked best-effort: OpenMVS's CLI flag names have changed
across releases and the exact InterfaceCOLMAP/TextureMesh invocation was not
executable-tested against a live OpenMVS install (unlike 07-10, which are
tested against this repo's real COLMAP data). Confirm flag names with
`--help` on your installed version before relying on this in a time crunch,
and see docs/troubleshooting.md.

Input:  <scene>/dense/  (COLMAP's undistorted workspace, from 08_undistort.py)
        <scene>/dense/meshed-poisson.ply  (from 10_fuse_mesh.py, reused as
        the input geometry so texturing runs on the same mesh, not a
        different one OpenMVS would otherwise reconstruct itself)
Output: <scene>/openmvs/textured.obj (+ texture PNG), output/11_texture_openmvs.json
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_common import DEFAULT_SCENE, run_stage, write_json


def find_openmvs_tool(name, bin_dir):
    if bin_dir:
        candidate = os.path.join(bin_dir, name)
        for c in (candidate, candidate + ".exe"):
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(f"{name} not found in --openmvs-bin-dir {bin_dir}")
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Could not find '{name}' on PATH. Pass --openmvs-bin-dir pointing at the "
        f"unzipped OpenMVS_Windows_x64_CUDA release, or add it to PATH. "
        f"See docs/setup_rog.md."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=DEFAULT_SCENE, help=f"Scene directory (default: {DEFAULT_SCENE})")
    parser.add_argument("--openmvs-bin-dir", default=None,
                         help="Directory containing InterfaceCOLMAP/ReconstructMesh/TextureMesh "
                              "(default: search PATH)")
    parser.add_argument("--mesh", default=None,
                         help="Input mesh to texture (default: <scene>/dense/meshed-poisson.ply "
                              "from 10_fuse_mesh.py)")
    parser.add_argument("--out", default=None, help="Output dir (default: <scene>/openmvs)")
    parser.add_argument("--dry-run", action="store_true", help="Print each command without running it")
    args = parser.parse_args()

    dense_dir = os.path.join(args.scene, "dense")
    mesh_path = args.mesh or os.path.join(dense_dir, "meshed-poisson.ply")
    out_dir = args.out or os.path.join(args.scene, "openmvs")

    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.isfile(mesh_path):
            parser.error(f"{mesh_path} not found -- run 10_fuse_mesh.py first, "
                         f"or pass --mesh explicitly")

    interface_colmap = find_openmvs_tool("InterfaceCOLMAP", args.openmvs_bin_dir)
    texture_mesh = find_openmvs_tool("TextureMesh", args.openmvs_bin_dir)

    scene_mvs = os.path.join(out_dir, "scene.mvs")
    textured_out = os.path.join(out_dir, "textured")

    run_stage(
        "11a_interface_colmap",
        [interface_colmap, "--input-file", dense_dir, "--output-file", scene_mvs],
        dry_run=args.dry_run,
    )
    result = run_stage(
        "11b_texture_mesh",
        [texture_mesh, scene_mvs, "--mesh-file", mesh_path,
         "-o", textured_out, "--export-type", "obj"],
        dry_run=args.dry_run,
    )

    write_json("11_texture_openmvs", {
        "scene": args.scene, "mesh_in": mesh_path, "out_dir": out_dir,
        "obj_out": textured_out + ".obj",
    })

    if not args.dry_run:
        print(f"\nIf TextureMesh's flags above didn't match your OpenMVS version "
              f"(check with `TextureMesh --help`), see docs/troubleshooting.md.")
        print("Next: python src/12_export_mesh.py --mesh", textured_out + ".obj")


if __name__ == "__main__":
    main()
