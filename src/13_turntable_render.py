"""
Stage 13 -- 360-degree turntable render of the final mesh, to MP4.

Because this pipeline recovers real 3D geometry (unlike the IBR track's
plane-proxy renderer, src/05_view_dependent_render.py, whose virtual camera
had to stay near the captured ~170 degree arc, or it would sweep off the
edge of a single flat plane), the camera path here is a full, arbitrary
orbit around the mesh -- a genuine upside of having recovered actual 3D
structure rather than synthesizing 2D views.

Rendering engine: Open3D's legacy `Visualizer(visible=False)`, NOT the
newer `visualization.rendering.OffscreenRenderer`. That was tried first and
fails on macOS with "EGL Headless is not supported on this platform" --
OffscreenRenderer's Filament backend needs EGL, which is Linux/Windows-only.
The legacy Visualizer uses an off-screen GLFW context instead, which was
verified working headlessly on this Mac. Frames are written through
cv2.VideoWriter, matching 05_view_dependent_render.py / 06_view_interpolation.py.

Input:  a mesh file (typically the .glb from 12_export_mesh.py)
Output: output/garuda_mesh_turntable.mp4 (or --out)
"""

import argparse
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_common import OUTPUT_DIR, ensure_output_dir

DEFAULT_OUT = os.path.join(OUTPUT_DIR, "garuda_mesh_turntable.mp4")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", required=True, help="Input mesh (.glb/.obj/.ply)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output video (default: {DEFAULT_OUT})")
    parser.add_argument("--frames", type=int, default=180, help="Frames for a full 360-degree orbit (default: 180)")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate (default: 30)")
    parser.add_argument("--width", type=int, default=960, help="Frame width (default: 960)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    parser.add_argument("--zoom", type=float, default=0.7,
                         help="Open3D camera zoom (smaller = closer/more zoomed in; default: 0.7)")
    args = parser.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        print("ERROR: open3d is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    if not os.path.isfile(args.mesh):
        print(f"ERROR: {args.mesh} not found")
        sys.exit(1)

    print(f"Loading {args.mesh}...")
    mesh = o3d.io.read_triangle_mesh(args.mesh, enable_post_processing=True)
    if len(mesh.vertices) == 0:
        print(f"ERROR: {args.mesh} loaded with 0 vertices -- check the file/format")
        sys.exit(1)
    mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors() and not mesh.has_textures():
        mesh.paint_uniform_color([0.75, 0.72, 0.68])  # neutral fallback so an untextured mesh isn't pure black

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(visible=False, width=args.width, height=args.height)
    if not ok:
        print("ERROR: Open3D could not create an off-screen render window on this machine.")
        print("See docs/troubleshooting.md -- as a fallback, open the mesh in Meshlab/Blender "
              "and record a turntable manually.")
        sys.exit(1)

    vis.add_geometry(mesh)
    render_opt = vis.get_render_option()
    render_opt.mesh_show_back_face = True
    render_opt.background_color = np.array([1.0, 1.0, 1.0])

    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(args.zoom)

    ensure_output_dir()
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))

    step_deg = 360.0 / args.frames
    # Open3D's rotate() takes pixel-ish units tied to the trackball, not
    # degrees directly; empirically ~width/2 units per full 360 rotation for
    # the default trackball sensitivity. Scale per-frame rotation off that.
    units_per_frame = (args.width / 2.0) / args.frames * (step_deg / (360.0 / args.frames))

    for i in range(args.frames):
        view_ctl.rotate(units_per_frame, 0.0)
        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_float_buffer(do_render=True)
        frame = (np.asarray(img) * 255).astype(np.uint8)  # RGB, float [0,1] -> uint8
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        if (i + 1) % 30 == 0 or i == args.frames - 1:
            print(f"  frame {i + 1}/{args.frames}")

    writer.release()
    vis.destroy_window()

    size_mb = round(os.path.getsize(args.out) / 1e6, 1)
    print(f"[saved] {args.out} ({size_mb} MB, {args.frames} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
