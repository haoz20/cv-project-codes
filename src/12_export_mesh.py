"""
Stage 12 -- Export the final mesh to a submission/viewing format.

Uses trimesh (handles both a vertex-coloured .ply from 10_fuse_mesh.py and a
UV-textured .obj+.mtl from 11_texture_openmvs.py generically) to convert to
.glb -- a single self-contained, web-viewable file. This resolves the
original plan's open "final export format" question: .glb for
viewing/submission, the source .ply/.obj kept as the raw/native backup.

Lightweight enough to run on the Mac once the mesh exists -- the ROG's job
ends at Stage 10/11.

Input:  a .ply or .obj mesh (from 10_fuse_mesh.py or 11_texture_openmvs.py)
Output: <same name>.glb (or --out), output/12_export_mesh.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", help="Input mesh (.ply or .obj)")
    parser.add_argument("--out", default=None, help="Output .glb path (default: <mesh> with .glb extension)")
    args = parser.parse_args()

    try:
        import trimesh
    except ImportError:
        print("ERROR: trimesh is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    if not os.path.isfile(args.mesh):
        print(f"ERROR: {args.mesh} not found")
        sys.exit(1)

    out_path = args.out or os.path.splitext(args.mesh)[0] + ".glb"

    print(f"Loading {args.mesh}...")
    mesh = trimesh.load(args.mesh, process=False)

    # trimesh.load can return a Scene (e.g. an .obj with an .mtl texture) or
    # a bare Trimesh (a vertex-coloured .ply). Normalize to report the same
    # stats either way.
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        n_vertices = sum(len(g.vertices) for g in geoms)
        n_faces = sum(len(g.faces) for g in geoms)
        has_texture = any(getattr(g.visual, "material", None) is not None for g in geoms)
        has_vertex_color = False
    else:
        n_vertices = len(mesh.vertices)
        n_faces = len(mesh.faces)
        has_texture = getattr(mesh.visual, "material", None) is not None
        has_vertex_color = hasattr(mesh.visual, "vertex_colors")

    print(f"  vertices: {n_vertices:,}   faces: {n_faces:,}")
    print(f"  has texture atlas: {has_texture}   has vertex colour: {has_vertex_color}")

    mesh.export(out_path)
    size_mb = round(os.path.getsize(out_path) / 1e6, 1)
    print(f"[saved] {out_path} ({size_mb} MB)")

    write_json("12_export_mesh", {
        "input": args.mesh, "output": out_path,
        "vertices": n_vertices, "faces": n_faces,
        "has_texture": has_texture, "has_vertex_color": has_vertex_color,
        "output_size_mb": size_mb,
    })

    print("\nNext: python src/13_turntable_render.py --mesh", out_path)


if __name__ == "__main__":
    main()
