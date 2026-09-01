"""
Stage 6 — View-dependent rendering (View-Dependent Texture Mapping track)

Based on Szeliski's "Computer Vision: Algorithms and Applications", Ch. 14 —
view-dependent texture mapping (§14.1.1, Debevec, Taylor and Malik 1996) and
unstructured lumigraph rendering (§14.3.1, Buehler et al. 2001).

Renders novel views of the Garuda by blending the original photographs at
render time, rather than by reconstructing a mesh or training a radiance
field. For each virtual camera along the capture arc:

  1. every source photo is warped into the virtual view through the plane
     proxy from 04_plane_proxy.py, using the plane-induced homography
         H = K_v (R_vi + t_vi n_i^T / d_i) K_i^-1
  2. each warped photo is weighted per pixel by angular closeness — the angle
     between the virtual view ray and that source camera's ray to the same
     point on the proxy — so the photos taken from nearest the virtual
     viewpoint dominate (weight ~ 1/angle, Debevec et al. 1996)
  3. the weighted photos are blended, and the frame is written to a video.

The virtual path sweeps the real capture arc, not a full 360 degree orbit:
the relief is wall-mounted, there is no back side, and no source photo exists
off the arc to blend from. Sweeping only what was actually captured is the
honest choice.

It sweeps the *centre* of that arc, though, not all of it. The photos span
167 degrees around the subject, which puts the outermost cameras within a few
degrees of the plane itself — and a planar proxy seen edge-on projects to a
sliver, so those frames degenerate into extreme skew no matter how well the
plane is fitted. That is a property of the proxy, not of the fit: a single
plane only supports viewpoints reasonably far from grazing. --arc-limit
therefore bounds the path to the range where the proxy holds (default +-55
degrees from head-on) while still blending from *every* photo, including the
ones outside that range. The printout states the swept span so the figure in
the report can be labelled with what was actually rendered.

Output: output/garuda_view_dependent_sweep.mp4 (plus a sample-frames PNG),
directly comparable to the Instant-NGP turntable in the Results section.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, REPO_ROOT, bgr2rgb, save_figure
from colmap_model import load_model

DEFAULT_MODEL = os.path.join(REPO_ROOT, "colmap_work", "sparse", "0")
DEFAULT_IMAGES = os.path.join(REPO_ROOT, "colmap_work", "images")
DEFAULT_PLANE = os.path.join(OUTPUT_DIR, "plane_proxy.json")
DEFAULT_VIDEO = os.path.join(OUTPUT_DIR, "garuda_view_dependent_sweep.mp4")

EPS = 1e-9


# --------------------------------------------------------------------------
# Source photo access
# --------------------------------------------------------------------------

class SourceImages:
    """
    Lazily loads, downscales and undistorts source photos, with a small cache.

    The 99 photos are 8064x6048 each — loading them all would be several GB.
    Only the handful of cameras nearest the current virtual view are needed
    per frame, and consecutive frames reuse most of them, so a small LRU
    cache keeps the whole render inside a few hundred MB.

    Undistortion matters here: the reconstruction used SIMPLE_RADIAL, and the
    plane homography is a pinhole model. Warping the raw (distorted) pixels
    would misalign the photos against each other towards the frame edges,
    which is exactly where blending artefacts are most visible.
    """

    def __init__(self, image_dir, cameras, max_dim, cache_size=24):
        self.image_dir = image_dir
        self.cameras = cameras
        self.max_dim = max_dim
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self._scales = {}

    def scale_for(self, camera_id):
        """Resize factor applied to photos from this camera."""
        if camera_id not in self._scales:
            cam = self.cameras[camera_id]
            self._scales[camera_id] = min(1.0, self.max_dim / max(cam.width, cam.height))
        return self._scales[camera_id]

    def K(self, camera_id):
        """Intrinsics of a source photo at the downscaled resolution."""
        return self.cameras[camera_id].K(self.scale_for(camera_id))

    def get(self, image):
        """Return the undistorted, downscaled BGR photo for a COLMAP Image."""
        if image.id in self._cache:
            self._cache.move_to_end(image.id)
            return self._cache[image.id]

        path = os.path.join(self.image_dir, image.name)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read source photo: {path}")

        cam = self.cameras[image.camera_id]
        scale = self.scale_for(image.camera_id)
        if scale < 1.0:
            img = cv2.resize(img, (int(round(cam.width * scale)), int(round(cam.height * scale))),
                             interpolation=cv2.INTER_AREA)

        K = self.K(image.camera_id)
        img = cv2.undistort(img, K, cam.dist_coeffs())

        self._cache[image.id] = img
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return img


# --------------------------------------------------------------------------
# Virtual camera path along the capture arc
# --------------------------------------------------------------------------

def plane_basis(normal, centers, centroid):
    """
    Orthonormal frame on the proxy: (u, v, normal).

    `u` is the in-plane direction the cameras spread out along, i.e. the
    direction the capture arc sweeps, so ordering the cameras by their angle
    in the (u, normal) plane recovers the order they were shot along the arc.
    """
    rel = centers - centroid
    tangential = rel - np.outer(rel @ normal, normal)
    _, _, vt = np.linalg.svd(tangential - tangential.mean(axis=0), full_matrices=False)
    u = vt[0] - (vt[0] @ normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v, normal


def order_along_arc(images, normal, centroid):
    """
    Sort the source cameras into the order they sweep across the arc.

    COLMAP image IDs follow the filename order, which is usually the shooting
    order — but not reliably, and a re-registered image can land anywhere. The
    geometry is the safer key: measure each camera's angle around the subject
    in the plane spanned by the proxy normal and the arc direction, and sort
    on that. Returns (sorted_images, angles_deg) with angles relative to the
    plane normal, so 0 deg is a head-on view.
    """
    centers = np.array([im.C for im in images])
    u, _, n = plane_basis(normal, centers, centroid)

    rel = centers - centroid
    along = rel @ u          # position across the arc
    height = rel @ n         # distance out from the plane
    angles = np.degrees(np.arctan2(along, height))

    order = np.argsort(angles)
    return [images[i] for i in order], angles[order]


def smooth_path(points, window):
    """
    Moving-average smoothing of the camera centres along the arc.

    Hand-held capture leaves the real camera centres jittery; interpolating
    straight through them makes the virtual camera visibly wobble. The ends
    are held rather than wrapped, since the arc is open, not a loop.
    """
    if window <= 1:
        return points.copy()

    half = window // 2
    padded = np.vstack([
        np.repeat(points[:1], half, axis=0),
        points,
        np.repeat(points[-1:], half, axis=0),
    ])
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(3)], axis=1)


def look_at(center, target, up_hint):
    """
    Build a world->camera rotation for a camera at `center` looking at `target`.

    Returns R in COLMAP's convention (X_cam = R @ X_world + t), with +z along
    the optical axis and +y *down* the image, matching how the source photos
    are posed. Note the sign: taking cross(up_hint, z) instead of
    cross(z, up_hint) builds a frame with +y pointing up the image and +x
    mirrored, i.e. every rendered frame upside down.
    """
    z = target - center
    z /= np.linalg.norm(z)

    x = np.cross(z, up_hint)
    if np.linalg.norm(x) < 1e-8:            # up_hint parallel to the view axis
        x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)

    y = np.cross(z, x)
    return np.stack([x, y, z])              # rows are the camera axes in world coords


def build_virtual_path(sorted_images, n_frames, smoothing, centroid, pingpong):
    """
    Sample a smooth virtual camera path across the capture arc.

    Resolves PLAN.md's open question on path parameterisation: the centres are
    interpolated along the *real* camera centres (so the virtual camera stays
    on the arc that was actually photographed, and every frame has genuinely
    nearby source views to blend), while the orientations are rebuilt as a
    look-at towards the subject centroid rather than interpolated from the
    source poses. Slerping the real rotations inherits every framing wobble of
    a hand-held capture; a look-at keeps the subject nailed to the centre of
    frame, which is what makes the sweep readable next to a NeRF turntable.

    Returns a list of (center, R) pairs.
    """
    centers = smooth_path(np.array([im.C for im in sorted_images]), smoothing)

    # A stable up vector: the mean of the source cameras' own up axes (-y in
    # COLMAP's convention, which points down), so the render keeps the same
    # horizon the photos were shot with.
    up = -np.mean([im.R[1] for im in sorted_images], axis=0)
    up /= np.linalg.norm(up)

    ts = np.linspace(0.0, len(centers) - 1.0, n_frames)
    if pingpong:
        ts = np.concatenate([ts, ts[-2:0:-1]])

    path = []
    for t in ts:
        i = int(np.clip(np.floor(t), 0, len(centers) - 2))
        frac = t - i
        center = centers[i] * (1.0 - frac) + centers[i + 1] * frac
        path.append((center, look_at(center, centroid, up)))
    return path


# --------------------------------------------------------------------------
# View-dependent blending
# --------------------------------------------------------------------------

def plane_homography(K_virt, R_virt, t_virt, K_src, R_src, t_src, normal, d):
    """
    Homography mapping source-image pixels to virtual-image pixels.

    For a plane n . X + d = 0 in world coordinates, expressed in source camera
    i's frame as n_i . X_i = d_i, the plane-induced homography is

        H = K_v (R_vi + t_vi n_i^T / d_i) K_i^-1

    with (R_vi, t_vi) the relative pose from source to virtual camera
    (Szeliski eq. 2.71 / 8.19).
    """
    n_i = R_src @ normal
    d_i = n_i @ t_src - d

    R_vi = R_virt @ R_src.T
    t_vi = t_virt - R_vi @ t_src

    H = K_virt @ (R_vi + np.outer(t_vi, n_i) / d_i) @ np.linalg.inv(K_src)
    return H


def plane_points(K_virt, R_virt, center, normal, d, width, height):
    """
    Intersect every virtual-camera pixel ray with the proxy plane.

    Returns (points, valid): the world-space point each pixel sees on the
    proxy, and a mask of the pixels whose ray actually hits it (rays parallel
    to the plane, or pointing away from it, do not).
    """
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    pixels = np.stack([us, vs, np.ones_like(us)], axis=-1)

    # Pixel rays in world coordinates.
    dirs = pixels @ np.linalg.inv(K_virt).T @ R_virt      # (R^T @ (K^-1 p))^T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    denom = dirs @ normal
    dist_to_plane = center @ normal + d                   # > 0: camera in front

    valid = denom < -EPS                                  # ray heading at the plane
    t = np.where(valid, -dist_to_plane / np.where(valid, denom, -1.0), 0.0)
    valid &= t > 0

    points = center + t[..., None] * dirs
    return points, valid


def select_sources(images, center, centroid, k, max_angle_deg):
    """
    Pick the k source photos closest in viewing angle to the virtual camera.

    Closeness is measured at the subject centroid — the angle between the
    virtual camera's ray to the centroid and each source camera's ray to it.
    This is the coarse, per-image version of the criterion; the fine,
    per-pixel version is what actually sets the blend weights below.
    """
    v_dir = centroid - center
    v_dir /= np.linalg.norm(v_dir)

    angles = []
    for im in images:
        s_dir = centroid - im.C
        s_dir /= np.linalg.norm(s_dir)
        angles.append(np.degrees(np.arccos(np.clip(v_dir @ s_dir, -1.0, 1.0))))
    angles = np.array(angles)

    order = np.argsort(angles)[:k]
    order = order[angles[order] <= max_angle_deg]
    return [images[i] for i in order], angles[order]


def blend_weights(angles, valid):
    """
    Unstructured-lumigraph blending field over the selected source photos.

    Weight falls off as 1/angle (Debevec, Taylor and Malik 1996), but a plain
    1/angle is discontinuous: as the virtual camera moves and a photo drops
    out of the k-nearest set, its weight vanishes abruptly and the frame
    flickers. Buehler et al. (2001) fix this by subtracting the weight of the
    furthest selected photo, so each photo's contribution decays to exactly
    zero at the moment it is about to be dropped:

        w_i = max(0, 1/theta_i - 1/theta_k)

    Parameters
    ----------
    angles : (K,H,W) per-pixel angles between virtual and source view rays.
    valid  : (K,H,W) bool, whether source k covers that pixel at all.

    Returns (K,H,W) weights summing to 1 over K wherever any source is valid.
    """
    inv = np.where(valid, 1.0 / (angles + EPS), 0.0)

    # 1/theta of the *worst* still-valid source, per pixel.
    inv_worst = np.min(np.where(valid, inv, np.inf), axis=0)
    inv_worst = np.where(np.isfinite(inv_worst), inv_worst, 0.0)

    w = np.clip(inv - inv_worst, 0.0, None)
    w = np.where(valid, w, 0.0)

    # Where only one source covers the pixel the subtraction zeroes it out;
    # fall back to raw 1/angle there so the pixel is still filled.
    total = w.sum(axis=0)
    fallback = total <= EPS
    if fallback.any():
        w = np.where(fallback[None, :, :], inv, w)
        total = w.sum(axis=0)

    return w, total


def render_frame(center, R_virt, K_virt, width, height, images, sources,
                 normal, d, centroid, k, max_angle_deg):
    """
    Render one novel view by warping and blending the nearest source photos.

    Returns (frame_bgr, coverage_mask, used_image_names).
    """
    t_virt = -R_virt @ center
    points, ray_valid = plane_points(K_virt, R_virt, center, normal, d, width, height)

    selected, _ = select_sources(images, center, centroid, k, max_angle_deg)
    if not selected:
        return np.zeros((height, width, 3), np.uint8), np.zeros((height, width), bool), []

    # Virtual view rays at each proxy point (shared across all sources).
    v_rays = points - center
    v_rays /= np.linalg.norm(v_rays, axis=-1, keepdims=True) + EPS

    warped, angles, valids = [], [], []
    for im in selected:
        src = sources.get(im)
        H = plane_homography(K_virt, R_virt, t_virt,
                             sources.K(im.camera_id), im.R, im.tvec, normal, d)

        warped.append(cv2.warpPerspective(src, H, (width, height), flags=cv2.INTER_LINEAR))

        # Which virtual pixels the source photo actually covers. Eroded by a
        # pixel so the warp's interpolated edge doesn't bleed into the blend.
        cover = cv2.warpPerspective(np.full(src.shape[:2], 255, np.uint8), H,
                                    (width, height), flags=cv2.INTER_NEAREST)
        cover = cv2.erode(cover, np.ones((3, 3), np.uint8))
        valids.append((cover > 0) & ray_valid)

        # Per-pixel angular closeness: virtual ray vs this camera's ray to the
        # same point on the proxy. This is the view-dependent part.
        s_rays = points - im.C
        s_rays /= np.linalg.norm(s_rays, axis=-1, keepdims=True) + EPS
        angles.append(np.arccos(np.clip(np.sum(v_rays * s_rays, axis=-1), -1.0, 1.0)))

    warped = np.stack(warped).astype(np.float32)
    angles = np.stack(angles)
    valids = np.stack(valids)

    w, total = blend_weights(angles, valids)
    covered = total > EPS

    blended = (w[..., None] * warped).sum(axis=0)
    blended /= np.where(covered, total, 1.0)[..., None]
    frame = np.where(covered[..., None], blended, 0.0)

    return np.clip(frame, 0, 255).astype(np.uint8), covered, [im.name for im in selected]


# --------------------------------------------------------------------------

def save_sample_frames(frames, angles, name, show):
    """Save start / middle / end frames as one figure for the report."""
    fig, axes = plt.subplots(1, len(frames), figsize=(5.5 * len(frames), 4.5))
    for ax, (frame, label) in zip(np.atleast_1d(axes), zip(frames, angles)):
        ax.imshow(bgr2rgb(frame))
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Stage 6 — view-dependent renders along the capture arc")
    fig.tight_layout()
    save_figure(name, show=show)


def main():
    parser = argparse.ArgumentParser(
        description="Render a view-dependent texture-mapped sweep across the capture arc."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="COLMAP sparse model folder (default: colmap_work/sparse/0)")
    parser.add_argument("--images", default=DEFAULT_IMAGES,
                        help="Folder of source photos (default: colmap_work/images)")
    parser.add_argument("--plane", default=DEFAULT_PLANE,
                        help="Plane proxy from 04 (default: output/plane_proxy.json)")
    parser.add_argument("--out", default=DEFAULT_VIDEO,
                        help="Output video (default: output/garuda_view_dependent_sweep.mp4)")
    parser.add_argument("--frames", type=int, default=180,
                        help="Frames across the arc (default: 180)")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate (default: 30)")
    parser.add_argument("--width", type=int, default=960,
                        help="Render width in px; height follows the camera aspect (default: 960)")
    parser.add_argument("--sources", type=int, default=4,
                        help="Source photos blended per frame. The photos sit a median 4.5 deg "
                             "apart, so blending many of them pulls in views far enough away "
                             "that proxy parallax shows up as ghosting; 4 is a good balance "
                             "between sharpness and smooth transitions (default: 4)")
    parser.add_argument("--max-dim", type=int, default=1600,
                        help="Downscale source photos to this longest side (default: 1600)")
    parser.add_argument("--max-angle", type=float, default=45.0,
                        help="Ignore source photos more than this many degrees from the "
                             "virtual view (default: 45)")
    parser.add_argument("--arc-limit", type=float, default=55.0,
                        help="Bound the virtual path to this many degrees either side of "
                             "head-on. Beyond roughly this angle the virtual camera looks "
                             "along the proxy plane and the render degenerates. Source photos "
                             "outside the range are still blended (default: 55)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Moving-average window over the camera path, in source "
                             "cameras; 1 disables (default: 5)")
    parser.add_argument("--min-plane-dist", type=float, default=0.1,
                        help="Drop source cameras closer to the proxy than this fraction of "
                             "the median camera distance — their homographies are "
                             "ill-conditioned (default: 0.1)")
    parser.add_argument("--pingpong", action="store_true",
                        help="Sweep the arc and return, so the video loops seamlessly")
    parser.add_argument("--show", action="store_true",
                        help="Also display the sample-frames figure interactively")
    args = parser.parse_args()

    with open(args.plane) as f:
        plane = json.load(f)
    normal = np.array(plane["normal"], dtype=np.float64)
    d = float(plane["d"])
    centroid = np.array(plane["centroid"], dtype=np.float64)

    cameras, images_by_id, _ = load_model(args.model)
    images = sorted(images_by_id.values(), key=lambda im: im.id)

    # Cameras sitting almost on the proxy give d_i ~ 0 in the homography.
    dists = np.array([abs(im.C @ normal + d) for im in images])
    keep = dists > args.min_plane_dist * np.median(dists)
    if (~keep).any():
        dropped = [im.name for im, k in zip(images, keep) if not k]
        print(f"Dropping {len(dropped)} source photo(s) too close to the proxy plane: "
              f"{', '.join(dropped)}")
    images = [im for im, k in zip(images, keep) if k]

    sorted_images, arc_angles = order_along_arc(images, normal, centroid)
    arc_span = arc_angles.max() - arc_angles.min()

    # The path is bounded to where the plane proxy is well conditioned; the
    # blend still draws on every photo, including those outside the bound.
    on_path = np.abs(arc_angles) <= args.arc_limit
    if on_path.sum() < 2:
        raise SystemExit(f"--arc-limit {args.arc_limit} leaves fewer than 2 cameras on the path")
    path_images = [im for im, k in zip(sorted_images, on_path) if k]
    path_angles = arc_angles[on_path]

    cam = cameras[sorted_images[0].camera_id]
    width = args.width
    height = int(round(width * cam.height / cam.width))
    K_virt = cam.K(width / cam.width)

    print(f"Plane proxy : {args.plane}")
    print(f"Source views: {len(sorted_images)} photos spanning {arc_span:.1f} deg of arc "
          f"({arc_angles.min():.1f} to {arc_angles.max():.1f} deg from head-on)")
    print(f"Virtual path: {path_angles.min():.1f} to {path_angles.max():.1f} deg "
          f"({path_angles.max() - path_angles.min():.1f} deg swept, "
          f"--arc-limit {args.arc_limit:g}), guided by {len(path_images)} cameras")
    print(f"Render      : {width}x{height}, {args.sources} sources/frame, "
          f"sources downscaled to {args.max_dim}px\n")

    path = build_virtual_path(path_images, args.frames, args.smooth, centroid, args.pingpong)
    sources = SourceImages(args.images, cameras, args.max_dim)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {args.out}")

    sample_idx = {
        0: f"arc start ({path_angles.min():.0f} deg)",
        len(path) // 2: "arc middle (head-on)",
        len(path) - 1: f"arc end ({path_angles.max():.0f} deg)",
    }
    samples, sample_labels = [], []
    coverages = []

    try:
        for i, (center, R_virt) in enumerate(path):
            frame, covered, used = render_frame(
                center, R_virt, K_virt, width, height, sorted_images, sources,
                normal, d, centroid, args.sources, args.max_angle,
            )
            writer.write(frame)
            coverages.append(covered.mean())

            if i in sample_idx:
                samples.append(frame)
                sample_labels.append(f"{sample_idx[i]} — {len(used)} photos blended")

            if i % 20 == 0 or i == len(path) - 1:
                print(f"  frame {i + 1:4d}/{len(path)}  "
                      f"sources={len(used):2d}  coverage={covered.mean() * 100:5.1f}%")
    finally:
        writer.release()

    print(f"\n[saved] {args.out}")
    print(f"  {len(path)} frames at {args.fps} fps ({len(path) / args.fps:.1f} s), "
          f"mean coverage {np.mean(coverages) * 100:.1f}%")

    if samples:
        save_sample_frames(samples, sample_labels, "05_view_dependent_frames", args.show)


if __name__ == "__main__":
    main()
