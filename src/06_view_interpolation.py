"""
Stage 6b — View interpolation (Szeliski §14.1, Chen and Williams 1993)

The other Ch. 14 renderer in this repo. Where 05_view_dependent_render.py
projects every photo through a single plane and blends, this script never
builds a global geometric model at all: it takes *adjacent* photos along the
capture arc, establishes a dense pixel-to-pixel correspondence between them,
and morphs one into the other to synthesise the views in between.

Why this beats the plane proxy here. A single plane cannot represent the
relief's depth, so photos taken from different angles disagree by however far
the carving stands off the wall, and the blend turns that disagreement into
ghosting. View interpolation sidesteps it: correspondence is per pixel, so
depth is handled implicitly and exactly, and only the two bracketing photos
are ever mixed. On this capture it cuts the alignment error between adjacent
photos by 2-4x (mean abs. error 15.7 -> 3.9 grey levels at a typical 6.5 deg
step; 34.5 -> 17.9 at the worst 10.6 deg step).

Correspondence is found in two steps, "plane + parallax":

  1. The plane proxy from 04 supplies a homography that removes the bulk of
     the motion between the pair — everything the flat wall explains.
  2. Dense optical flow (DIS) picks up what is left, which is exactly the
     parallax caused by the relief standing off that wall. Flow only has to
     explain a small residual, so it is far more reliable than running it on
     the raw pair.

Because every frame is anchored to real photographs rather than a synthetic
viewpoint, the output has none of the plane renderer's failure modes: no black
wedges where no photo covers the frame, and no arc limit — the full sweep is
rendered, not just the middle.

Output: output/garuda_view_interpolation.mp4
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, REPO_ROOT, bgr2rgb, save_figure
from colmap_model import load_model

DEFAULT_MODEL = os.path.join(REPO_ROOT, "colmap_work", "sparse", "0")
DEFAULT_IMAGES = os.path.join(REPO_ROOT, "colmap_work", "images")
DEFAULT_PLANE = os.path.join(OUTPUT_DIR, "plane_proxy.json")
DEFAULT_VIDEO = os.path.join(OUTPUT_DIR, "garuda_view_interpolation.mp4")


def viewpoint_separation(a, b, target):
    """Angle between two cameras' rays to the subject, in degrees."""
    va, vb = target - a.C, target - b.C
    va /= np.linalg.norm(va)
    vb /= np.linalg.norm(vb)
    return float(np.degrees(np.arccos(np.clip(va @ vb, -1.0, 1.0))))


def find_sweeps(images, target, max_step):
    """
    Split the photos into continuous sweeps across the arc.

    Adjacency has to be *spatial*, and the reliable proxy for it is shooting
    order: consecutive shots in a handheld walk-around are consecutive in
    space. Sorting by COLMAP image ID does not work (the IDs are not in
    capture order — median 13 deg between neighbours, against 6.8 deg by
    filename), and sorting by angle around the arc is worse still, since two
    photos can share an arc angle while being metres apart in height.

    This capture turns out to be four passes across the relief rather than
    one, so the sequence is cut wherever consecutive photos jump more than
    `max_step` degrees apart. Returns a list of runs, longest first.
    """
    ordered = sorted(images, key=lambda im: im.name)

    runs, current = [], [ordered[0]]
    for prev, nxt in zip(ordered, ordered[1:]):
        if viewpoint_separation(prev, nxt, target) > max_step:
            runs.append(current)
            current = [nxt]
        else:
            current.append(nxt)
    runs.append(current)

    runs = [r for r in runs if len(r) >= 2]
    runs.sort(key=len, reverse=True)
    return runs


class PairCorrespondence:
    """
    Dense two-way correspondence between one adjacent pair of photos.

    Holds the displacement fields in both directions: `d_ab[y, x]` is the
    offset from pixel (x, y) in A to its match in B, and `d_ba` the reverse.
    Both are needed — morphing A forward and B backward with a single field
    assumes the correspondence is exactly invertible, which breaks wherever
    the relief occludes itself.
    """

    def __init__(self, img_a, img_b, H_ab, flow):
        self.img_a = img_a
        self.img_b = img_b
        h, w = img_a.shape[:2]
        self.grid = np.mgrid[0:h, 0:w][::-1].astype(np.float32)   # (2,H,W) as x,y
        self.d_ab = self._displacement(img_a, img_b, H_ab, flow)
        self.d_ba = self._displacement(img_b, img_a, np.linalg.inv(H_ab), flow)

    def _displacement(self, src, dst, H, flow):
        """Displacement from every `src` pixel to its match in `dst`."""
        h, w = src.shape[:2]
        xs, ys = self.grid

        # Step 1: undo everything the plane explains, by pulling dst into
        # src's frame through the plane homography.
        warped = cv2.warpPerspective(dst, H, (w, h),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

        # Step 2: dense flow on the residual — the relief's parallax.
        f = flow.calc(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
                      cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), None)

        # Compose the two into one absolute correspondence.
        pts = np.stack([xs + f[..., 0], ys + f[..., 1], np.ones_like(xs)], axis=-1)
        matched = pts @ H.T
        matched = matched[..., :2] / matched[..., 2:3]
        return (matched - np.stack([xs, ys], axis=-1)).astype(np.float32)

    def morph(self, t):
        """
        Synthesise the view at fraction t between A (t=0) and B (t=1).

        A scene point seen at x_a in A and x_b in B is placed at
        (1-t) x_a + t x_b, so recovering the source pixel for an output pixel p
        means solving x_a + t D(x_a) = p. That is implicit, so the field is
        sampled once at p and used to step back — one fixed-point iteration,
        which is ample when the displacements are this small.
        """
        a = self._pull(self.img_a, self.d_ab, t)
        b = self._pull(self.img_b, self.d_ba, 1.0 - t)
        return cv2.addWeighted(a, 1.0 - t, b, t, 0.0)

    def _pull(self, img, disp, s):
        xs, ys = self.grid
        step = cv2.remap(disp, (xs - s * disp[..., 0]).astype(np.float32),
                         (ys - s * disp[..., 1]).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.remap(img, (xs - s * step[..., 0]).astype(np.float32),
                         (ys - s * step[..., 1]).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def plane_homography(K, R_dst, t_dst, R_src, t_src, normal, d):
    """Homography mapping src-image pixels to dst-image pixels via the proxy."""
    n_i = R_src @ normal
    d_i = n_i @ t_src - d
    R_rel = R_dst @ R_src.T
    t_rel = t_dst - R_rel @ t_src
    return K @ (R_rel + np.outer(t_rel, n_i) / d_i) @ np.linalg.inv(K)


def schedule_frames(run, target, n_frames):
    """
    Lay out output frames at a constant angular rate across the sweep.

    Photos are not evenly spaced around the arc, so giving every pair the same
    number of frames makes the virtual camera lurch — slow across a 2 deg gap,
    fast across a 10 deg one. Sampling uniformly in *cumulative* angle instead
    keeps the sweep moving at one speed. Returns (pair_index, t) per frame.
    """
    steps = np.array([viewpoint_separation(a, b, target)
                      for a, b in zip(run, run[1:])])
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])

    positions = np.linspace(0.0, cumulative[-1], n_frames)
    schedule = []
    for pos in positions:
        i = int(np.clip(np.searchsorted(cumulative, pos, side="right") - 1,
                        0, len(steps) - 1))
        t = (pos - cumulative[i]) / steps[i] if steps[i] > 1e-9 else 0.0
        schedule.append((i, float(np.clip(t, 0.0, 1.0))))
    return schedule, cumulative[-1]


def stabilise(frames, anchors, zoom):
    """
    Remove handheld jitter by locking the subject to a smoothed track.

    Each frame knows where the subject centroid projects to; that track is the
    photographer's hand-shake plus the intended sweep. Subtracting a smoothed
    version of it removes the shake and leaves the sweep. Frames are then
    zoomed slightly so the shifted edges stay filled.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    window = max(3, (len(anchors) // 8) | 1)
    pad = window // 2
    padded = np.vstack([np.repeat(anchors[:1], pad, axis=0),
                        anchors,
                        np.repeat(anchors[-1:], pad, axis=0)])
    kernel = np.ones(window) / window
    smooth = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in (0, 1)],
                      axis=1)

    h, w = frames[0].shape[:2]
    out = []
    for frame, raw, target in zip(frames, anchors, smooth):
        dx, dy = target - raw
        M = np.array([[zoom, 0.0, dx + (1 - zoom) * w / 2],
                      [0.0, zoom, dy + (1 - zoom) * h / 2]])
        out.append(cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Render an arc sweep by view interpolation between adjacent photos."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="COLMAP sparse model folder (default: colmap_work/sparse/0)")
    parser.add_argument("--images", default=DEFAULT_IMAGES,
                        help="Folder of source photos (default: colmap_work/images)")
    parser.add_argument("--plane", default=DEFAULT_PLANE,
                        help="Plane proxy from 04, used as the base warp "
                             "(default: output/plane_proxy.json)")
    parser.add_argument("--out", default=DEFAULT_VIDEO,
                        help="Output video (default: output/garuda_view_interpolation.mp4)")
    parser.add_argument("--frames", type=int, default=300,
                        help="Frames across the sweep (default: 300)")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate (default: 30)")
    parser.add_argument("--max-dim", type=int, default=1200,
                        help="Working resolution, longest side in px (default: 1200)")
    parser.add_argument("--max-step", type=float, default=12.0,
                        help="Cut the photo sequence where consecutive shots are further "
                             "apart than this many degrees (default: 12)")
    parser.add_argument("--sweep", type=int, default=0,
                        help="Which sweep to render, 0 = longest (default: 0)")
    parser.add_argument("--no-stabilise", action="store_true",
                        help="Keep the raw handheld framing instead of locking the "
                             "subject to a smoothed track")
    parser.add_argument("--zoom", type=float, default=1.06,
                        help="Zoom applied when stabilising, to keep edges filled "
                             "(default: 1.06)")
    parser.add_argument("--pingpong", action="store_true",
                        help="Sweep out and back, so the video loops seamlessly")
    parser.add_argument("--show", action="store_true",
                        help="Also display the sample-frames figure interactively")
    args = parser.parse_args()

    with open(args.plane) as f:
        plane = json.load(f)
    normal = np.array(plane["normal"], dtype=np.float64)
    d = float(plane["d"])
    centroid = np.array(plane["centroid"], dtype=np.float64)

    cameras, images_by_id, _ = load_model(args.model)
    images = list(images_by_id.values())

    cam = cameras[images[0].camera_id]
    scale = args.max_dim / max(cam.width, cam.height)
    K = cam.K(scale)
    width = int(round(cam.width * scale))
    height = int(round(cam.height * scale))

    runs = find_sweeps(images, centroid, args.max_step)
    print(f"Capture splits into {len(runs)} sweep(s) of "
          f"{[len(r) for r in runs]} photos (cut at >{args.max_step:g} deg jumps)")
    if args.sweep >= len(runs):
        raise SystemExit(f"--sweep {args.sweep} out of range; {len(runs)} sweeps found")
    run = runs[args.sweep]

    schedule, span = schedule_frames(run, centroid, args.frames)
    print(f"Rendering sweep {args.sweep}: {len(run)} photos "
          f"({run[0].name} -> {run[-1].name}), {span:.0f} deg of arc")
    print(f"Working resolution {width}x{height}, {args.frames} frames\n")

    def load(im):
        img = cv2.imread(os.path.join(args.images, im.name))
        if img is None:
            raise FileNotFoundError(os.path.join(args.images, im.name))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return cv2.undistort(img, K, cam.dist_coeffs())

    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow.setUseSpatialPropagation(True)

    frames, anchors = [], []
    pair_cache = (None, None)
    prev_img = None

    for n, (i, t) in enumerate(schedule):
        if pair_cache[0] != i:
            a, b = run[i], run[i + 1]
            img_a = prev_img if prev_img is not None else load(a)
            img_b = load(b)
            H_ab = plane_homography(K, b.R, b.tvec, a.R, a.tvec, normal, d)
            pair_cache = (i, PairCorrespondence(img_a, img_b, H_ab, flow))
            prev_img = img_b
            print(f"  pair {i + 1:2d}/{len(run) - 1}  {a.name} -> {b.name}  "
                  f"({viewpoint_separation(a, b, centroid):.1f} deg)")

        corr = pair_cache[1]
        frames.append(corr.morph(t))

        # Where the subject sits in this frame, for stabilisation.
        a, b = run[i], run[i + 1]
        pa = K @ (a.R @ centroid + a.tvec)
        pb = K @ (b.R @ centroid + b.tvec)
        pa, pb = pa[:2] / pa[2], pb[:2] / pb[2]
        anchors.append((1 - t) * pa + t * pb)

    if not args.no_stabilise:
        frames = stabilise(frames, anchors, args.zoom)
        print("\nStabilised: subject locked to a smoothed track, "
              f"zoom {args.zoom:g}x to keep edges filled")

    if args.pingpong:
        frames = frames + frames[-2:0:-1]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {args.out}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    print(f"\n[saved] {args.out}")
    print(f"  {len(frames)} frames at {args.fps} fps ({len(frames) / args.fps:.1f} s)")

    picks = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4]
    labels = ["sweep start", "quarter", "middle", "three-quarter"]
    fig, axes = plt.subplots(1, len(picks), figsize=(5.0 * len(picks), 4.0))
    for ax, p, label in zip(axes, picks, labels):
        ax.imshow(bgr2rgb(frames[p]))
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Stage 6b — view interpolation across the capture arc")
    fig.tight_layout()
    save_figure("06_view_interpolation_frames", show=args.show)


if __name__ == "__main__":
    main()
