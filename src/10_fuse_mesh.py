"""
Stage 10 -- Fuse depth maps into a dense point cloud, then Poisson-mesh it.

Wraps `colmap stereo_fusion` (-> fused.ply, a dense point cloud with
per-point colour already sampled from the source photos and per-point
normals) followed by `colmap poisson_mesher` (-> meshed-poisson.ply). COLMAP's
Poisson mesher propagates per-vertex colour from the input cloud, so this
already produces a coloured mesh -- the safe, always-works texturing
baseline described in the plan; 11_texture_openmvs.py is the optional
UV-texture-atlas upgrade.

Also computes and logs the numbers the paper's Results section needs: sparse
vs. dense point count (the actual value MVS adds over SfM alone), mesh
vertex/face counts, and per-substage timing -- directly answering the
original plan's open question ("note actual processing time... for the
Results/Analysis section").

Expect a hole or noise above the eaves: every elevation band is still an
eye-level-and-up walkaround, so nothing looked straight down onto the top of
the relief. That's a capture-coverage limitation, not a bug -- report it,
don't hide it.

Input:  <scene>/dense/  (from 09_dense_stereo.py)
Output: <scene>/dense/fused.ply, <scene>/dense/meshed-poisson.ply,
        output/10_fuse_mesh.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_common import DEFAULT_SCENE, REPO_ROOT, find_binary, read_ply_header, run_stage, write_json
from colmap_model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=DEFAULT_SCENE, help=f"Scene directory (default: {DEFAULT_SCENE})")
    parser.add_argument("--colmap-bin", default=None, help="Path to colmap executable (default: search PATH / COLMAP_BIN)")
    parser.add_argument("--sparse", default=None,
                         help="Sparse model dir for the point-count comparison "
                              "(default: <scene>/sparse/0)")
    parser.add_argument("--dry-run", action="store_true", help="Print each command without running it")
    args = parser.parse_args()

    colmap_bin = args.colmap_bin or find_binary("colmap", "COLMAP_BIN")
    workspace_path = os.path.join(args.scene, "dense")
    fused_path = os.path.join(workspace_path, "fused.ply")
    mesh_path = os.path.join(workspace_path, "meshed-poisson.ply")
    sparse_dir = args.sparse or os.path.join(args.scene, "sparse", "0")

    if not args.dry_run and not os.path.isdir(workspace_path):
        parser.error(f"{workspace_path} not found -- run 09_dense_stereo.py first")

    fuse_log = run_stage(
        "10a_stereo_fusion",
        [colmap_bin, "stereo_fusion", "--workspace_path", workspace_path, "--output_path", fused_path],
        dry_run=args.dry_run,
    )
    mesh_log = run_stage(
        "10b_poisson_mesher",
        [colmap_bin, "poisson_mesher", "--input_path", fused_path, "--output_path", mesh_path],
        dry_run=args.dry_run,
    )

    result = {
        "scene": args.scene,
        "fused_ply": fused_path,
        "mesh_ply": mesh_path,
        "fusion_seconds": fuse_log["elapsed_seconds"],
        "poisson_seconds": mesh_log["elapsed_seconds"],
    }

    if not args.dry_run:
        n_sparse = None
        if os.path.isfile(os.path.join(sparse_dir, "points3D.txt")):
            _cams, _imgs, points3D = load_model(sparse_dir)
            n_sparse = int(points3D["xyz"].shape[0])

        fused_info = read_ply_header(fused_path)
        mesh_info = read_ply_header(mesh_path)

        result.update({
            "sparse_points": n_sparse,
            "fused_points": fused_info["vertex_count"],
            "fused_has_color": fused_info["has_color"],
            "densification_ratio": (
                round(fused_info["vertex_count"] / n_sparse, 2) if n_sparse else None
            ),
            "mesh_vertices": mesh_info["vertex_count"],
            "mesh_faces": mesh_info["face_count"],
            "mesh_has_color": mesh_info["has_color"],
            "mesh_file_size_mb": round(os.path.getsize(mesh_path) / 1e6, 1),
        })

        print("\n--- Dense reconstruction stats (for the paper's Results section) ---")
        if n_sparse:
            print(f"  sparse points (SfM):       {n_sparse:,}")
        print(f"  fused points (MVS):        {result['fused_points']:,}")
        if result["densification_ratio"]:
            print(f"  densification:             {result['densification_ratio']}x")
        print(f"  mesh vertices:              {result['mesh_vertices']:,}")
        print(f"  mesh faces:                 {result['mesh_faces']:,}")
        print(f"  mesh has vertex colour:     {result['mesh_has_color']}")
        print(f"  mesh file size:             {result['mesh_file_size_mb']} MB")
        print("\nNote: a hole or noisy region above the eaves is expected -- report it as a "
              "capture-coverage limitation, not a failure.")
        print("\nNext: python src/12_export_mesh.py --mesh", mesh_path,
              "  (or src/11_texture_openmvs.py first for a real UV texture atlas)")

    write_json("10_fuse_mesh", result)


if __name__ == "__main__":
    main()
