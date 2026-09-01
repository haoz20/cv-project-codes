"""
Stage 5 — Plane proxy fitting (View-Dependent Texture Mapping track)

Based on Szeliski's "Computer Vision: Algorithms and Applications", Ch. 14
(§14.1.1 view-dependent texture maps, §14.3.1 unstructured lumigraph).

Fits a single dominant plane to the sparse 3D point cloud produced by COLMAP.
That plane is the geometric proxy the source photos get projected through in
06 — the same choice Photo Tourism (Snavely, Seitz and Szeliski 2006) makes,
and a good fit for this subject: the Garuda is a wall-mounted relief, close
to planar, so no mesh reconstruction or MVS is needed.

Method, in two stages:

  1. RANSAC, for robustness against stray points (the sparse cloud has plenty
     of background/floor points nowhere near the relief), followed by an SVD
     refit on the inlier set — the smallest-singular-vector of the
     mean-centred inliers is the plane normal.

  2. A second SVD refit over every point within a band of that plane. Stage 1
     on its own locks onto the single thinnest, densest surface, which for
     this capture is the flat wall *behind* the relief; the carving itself
     then sits entirely in front of the proxy. Stage 2 re-fits through the
     middle of the carved slab instead, which is what the proxy should
     approximate. Depth error against the proxy is exactly what shows up as
     ghosting in the blended render, so this matters: on our model it drops
     the RMS residual over the subject from 0.28 to 0.17 COLMAP units for a
     3.5 degree change in orientation.

Writes output/plane_proxy.json, consumed by 05_view_dependent_render.py.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, REPO_ROOT, save_figure
from colmap_model import load_model

DEFAULT_MODEL = os.path.join(REPO_ROOT, "colmap_work", "sparse", "0")
DEFAULT_OUT = os.path.join(OUTPUT_DIR, "plane_proxy.json")


def fit_plane_svd(xyz):
    """
    Least-squares plane through a point set.

    Returns (normal, d) for the plane n . X + d = 0, with n a unit vector.
    The normal is the singular vector of the mean-centred points with the
    smallest singular value, i.e. the direction of least variance.
    """
    centroid = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - centroid, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    return normal, float(-normal @ centroid)


def fit_plane_ransac(xyz, threshold, iterations, rng):
    """
    RANSAC plane fit, refined by an SVD refit on the winning inlier set.

    Parameters
    ----------
    xyz : (N,3) array of 3D points.
    threshold : float
        Point-to-plane distance (in COLMAP world units) below which a point
        counts as an inlier.
    iterations : int
        Number of random 3-point hypotheses to try.

    Returns (normal, d, inlier_mask).
    """
    n_points = len(xyz)
    best_mask = None
    best_count = -1

    for _ in range(iterations):
        idx = rng.choice(n_points, size=3, replace=False)
        p0, p1, p2 = xyz[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:          # degenerate (collinear) sample
            continue
        normal = normal / norm
        d = -normal @ p0

        mask = np.abs(xyz @ normal + d) < threshold
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask

    if best_mask is None or best_count < 3:
        raise RuntimeError("RANSAC failed to find a plane — try a larger --threshold")

    # Refit on all inliers: the 3-point hypothesis only located the plane,
    # the SVD fit is what makes it accurate.
    normal, d = fit_plane_svd(xyz[best_mask])
    mask = np.abs(xyz @ normal + d) < threshold
    return normal, d, mask


def refit_plane_to_slab(xyz, normal, d, band):
    """
    Re-fit the plane through every point within `band` of it.

    Stage 2 of the fit (see the module docstring). The RANSAC plane supplies
    a robust orientation and picks out which points belong to the subject;
    this refit then uses that whole slab of points, so the plane passes
    through the middle of the relief rather than clinging to its rear face.

    Returns (normal, d, slab_mask). The normal keeps stage 1's orientation.
    """
    slab_mask = np.abs(xyz @ normal + d) < band
    if slab_mask.sum() < 3:
        raise RuntimeError("Refit band contains too few points — raise --refit-band")

    new_normal, new_d = fit_plane_svd(xyz[slab_mask])
    if new_normal @ normal < 0:      # keep stage 1's sign convention
        new_normal, new_d = -new_normal, -new_d
    return new_normal, new_d, slab_mask


def orient_towards_cameras(normal, d, camera_centers):
    """
    Flip the plane normal so it points at the cameras.

    A plane fit has no inherent sign; 05 relies on the normal facing the
    capture arc when it builds the plane-induced homographies.
    """
    signed = camera_centers @ normal + d
    if np.median(signed) < 0:
        return -normal, -d
    return normal, d


def sanity_check(normal, d, images, verbose=True):
    """
    Verify the fitted plane against the capture geometry, as PLAN.md requires.

    Two things have to hold for the proxy to be usable:
      1. the cameras all sit on one side of the plane (they were all in front
         of the wall, none behind it), and
      2. the cameras are looking *at* the plane — their optical axes should
         oppose the (camera-facing) plane normal.

    Returns a dict of the numbers, so they land in the JSON as well as stdout.
    """
    centers = np.array([im.C for im in images])
    view_dirs = np.array([im.viewing_direction for im in images])

    signed = centers @ normal + d
    frac_front = float((signed > 0).mean())

    # Angle between each optical axis and -n; 0 deg means dead-on to the plane.
    cosines = np.clip(view_dirs @ (-normal), -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))

    stats = {
        "cameras": len(images),
        "fraction_in_front": frac_front,
        "camera_distance_median": float(np.median(np.abs(signed))),
        "camera_distance_min": float(np.abs(signed).min()),
        "camera_distance_max": float(np.abs(signed).max()),
        "view_angle_to_plane_median_deg": float(np.median(angles)),
        "view_angle_to_plane_max_deg": float(angles.max()),
    }

    if verbose:
        print("Sanity check against capture geometry")
        print(f"  cameras in front of plane   : {frac_front * 100:.1f}% "
              f"({int(round(frac_front * len(images)))}/{len(images)})")
        print(f"  camera-plane distance       : median {stats['camera_distance_median']:.3f} "
              f"(min {stats['camera_distance_min']:.3f}, max {stats['camera_distance_max']:.3f})")
        print(f"  optical axis vs plane normal: median {stats['view_angle_to_plane_median_deg']:.1f} deg, "
              f"max {stats['view_angle_to_plane_max_deg']:.1f} deg")

        if frac_front < 0.95:
            print("  [warn] some cameras are behind the fitted plane — the proxy may be "
                  "fitting a background surface rather than the relief")
        if stats["view_angle_to_plane_median_deg"] > 60:
            print("  [warn] cameras are not facing the fitted plane — check the fit")

    return stats


def plot_diagnostics(xyz, ransac_mask, slab_mask, normal, d, centroid, images, name, show):
    """
    Two-panel figure: the fit seen edge-on, and the residual distribution.

    The left panel is drawn in a plane-aligned frame (in-plane spread on x,
    signed distance from the plane on y), so a good fit shows the subject
    collapsed into a horizontal band with the cameras arcing above it. The
    right panel is the check on stage 2: the residuals should straddle zero
    rather than sitting to one side of it.
    """
    centers = np.array([im.C for im in images])

    # Build an in-plane axis along the direction the cameras spread out.
    rel = centers - centroid
    tangential = rel - np.outer(rel @ normal, normal)
    _, _, vt = np.linalg.svd(tangential - tangential.mean(axis=0), full_matrices=False)
    u = vt[0]
    u = u - (u @ normal) * normal
    u /= np.linalg.norm(u)

    def to_2d(points):
        rel = points - centroid
        return rel @ u, rel @ normal

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    out_u, out_n = to_2d(xyz[~slab_mask])
    slab_u, slab_n = to_2d(xyz[slab_mask])
    core_u, core_n = to_2d(xyz[ransac_mask])
    cam_u, cam_n = to_2d(centers)
    ax.scatter(out_u, out_n, s=1, c="lightgray", label=f"outside band ({(~slab_mask).sum()})")
    ax.scatter(slab_u, slab_n, s=1, c="tab:blue", label=f"subject slab ({slab_mask.sum()})")
    ax.scatter(core_u, core_n, s=1, c="tab:green", label=f"RANSAC inliers ({ransac_mask.sum()})")
    ax.axhline(0.0, color="tab:red", lw=1.5, label="fitted plane (edge-on)")
    ax.scatter(cam_u, cam_n, s=18, c="tab:orange", marker="^", label=f"cameras ({len(centers)})")
    ax.set_xlabel("in-plane position (COLMAP units)")
    ax.set_ylabel("signed distance from plane")
    ax.set_title("Plane proxy, viewed edge-on")
    ax.legend(loc="upper right", fontsize=8, markerscale=4)
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1]
    dist = xyz[slab_mask] @ normal + d
    lim = np.percentile(np.abs(dist), 99)
    ax.hist(dist, bins=200, range=(-lim, lim), color="tab:blue")
    ax.axvline(0.0, color="tab:red", lw=1.5)
    ax.set_xlabel("signed point-plane distance (COLMAP units)")
    ax.set_ylabel("points")
    ax.set_title("Residuals of the subject slab about the plane")

    fig.suptitle("Stage 5 — dominant plane fitted to the COLMAP sparse cloud")
    fig.tight_layout()
    save_figure(name, show=show)


def main():
    parser = argparse.ArgumentParser(
        description="Fit a dominant plane proxy to the COLMAP sparse point cloud."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="COLMAP sparse model folder (default: colmap_work/sparse/0)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Where to write the fitted plane (default: output/plane_proxy.json)")
    parser.add_argument("--min-track", type=int, default=3,
                        help="Drop 3D points seen by fewer than this many images (default: 3)")
    parser.add_argument("--max-error", type=float, default=2.0,
                        help="Drop 3D points with mean reprojection error above this, px (default: 2.0)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="RANSAC inlier distance in COLMAP units "
                             "(default: 2%% of the median point-to-centroid distance)")
    parser.add_argument("--refit-band", type=float, default=0.5,
                        help="Stage-2 refit band, as a fraction of the scene scale. Every point "
                             "within this distance of the RANSAC plane is re-fitted, so the proxy "
                             "passes through the middle of the relief instead of its rear face. "
                             "0 disables stage 2 (default: 0.5)")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="RANSAC hypotheses to try (default: 2000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RANSAC random seed, for a reproducible fit (default: 0)")
    parser.add_argument("--show", action="store_true",
                        help="Also display the diagnostic figure interactively")
    args = parser.parse_args()

    cameras, images, points = load_model(
        args.model, min_track=args.min_track, max_error=args.max_error
    )
    xyz = points["xyz"]
    if len(xyz) < 3:
        raise SystemExit(f"Only {len(xyz)} points survived filtering — relax --min-track/--max-error")

    image_list = sorted(images.values(), key=lambda im: im.id)
    print(f"Model: {args.model}")
    print(f"  cameras (intrinsics)   : {len(cameras)}")
    print(f"  registered images      : {len(image_list)}")
    print(f"  3D points after filter : {len(xyz)} "
          f"(track >= {args.min_track}, error <= {args.max_error} px)\n")

    # Scene scale is arbitrary in an uncalibrated SfM reconstruction, so the
    # RANSAC threshold is derived from the cloud itself rather than hard-coded.
    centroid_all = xyz.mean(axis=0)
    scene_scale = float(np.median(np.linalg.norm(xyz - centroid_all, axis=1)))
    threshold = args.threshold if args.threshold is not None else 0.02 * scene_scale
    print(f"Scene scale (median radius): {scene_scale:.3f} COLMAP units")
    print(f"RANSAC inlier threshold    : {threshold:.4f}\n")

    rng = np.random.default_rng(args.seed)
    ransac_normal, ransac_d, ransac_mask = fit_plane_ransac(
        xyz, threshold, args.iterations, rng
    )
    print(f"Stage 1 (RANSAC): {ransac_mask.sum()}/{len(xyz)} inliers "
          f"({ransac_mask.mean() * 100:.1f}%)")

    if args.refit_band > 0:
        band = args.refit_band * scene_scale
        normal, d, slab_mask = refit_plane_to_slab(xyz, ransac_normal, ransac_d, band)
        tilt = np.degrees(np.arccos(np.clip(normal @ ransac_normal, -1.0, 1.0)))
        before = xyz[slab_mask] @ ransac_normal + ransac_d
        after = xyz[slab_mask] @ normal + d
        print(f"Stage 2 (refit)  : {slab_mask.sum()} points within {band:.3f} of the RANSAC plane")
        print(f"  orientation change    : {tilt:.2f} deg")
        print(f"  RMS residual over slab: {np.sqrt((before ** 2).mean()):.4f} -> "
              f"{np.sqrt((after ** 2).mean()):.4f} COLMAP units")
        print(f"  95th pct |residual|   : {np.percentile(np.abs(before), 95):.4f} -> "
              f"{np.percentile(np.abs(after), 95):.4f}\n")
    else:
        normal, d, slab_mask = ransac_normal, ransac_d, ransac_mask
        print("Stage 2 (refit)  : disabled (--refit-band 0)\n")

    centers = np.array([im.C for im in image_list])
    normal, d = orient_towards_cameras(normal, d, centers)

    residuals = xyz[slab_mask] @ normal + d
    centroid = xyz[slab_mask].mean(axis=0)

    print("Fitted plane  (n . X + d = 0, normal pointing towards the cameras)")
    print(f"  normal   : [{normal[0]: .5f}, {normal[1]: .5f}, {normal[2]: .5f}]")
    print(f"  d        : {d: .5f}")
    print(f"  centroid : [{centroid[0]: .4f}, {centroid[1]: .4f}, {centroid[2]: .4f}]")
    print(f"  points on proxy : {slab_mask.sum()}/{len(xyz)} ({slab_mask.mean() * 100:.1f}%)")
    print(f"  RMS residual    : {np.sqrt((residuals ** 2).mean()):.4f} COLMAP units")
    print(f"  95th pct        : {np.percentile(np.abs(residuals), 95):.4f} COLMAP units\n")

    stats = sanity_check(normal, d, image_list)

    plane = {
        "normal": normal.tolist(),
        "d": d,
        "centroid": centroid.tolist(),
        "convention": "n . X + d = 0; normal points from the plane towards the cameras",
        "fit": {
            "model": args.model,
            "points_used": int(len(xyz)),
            "ransac_inliers": int(ransac_mask.sum()),
            "ransac_threshold": threshold,
            "ransac_iterations": args.iterations,
            "refit_band": args.refit_band,
            "points_on_proxy": int(slab_mask.sum()),
            "seed": args.seed,
            "min_track": args.min_track,
            "max_error": args.max_error,
            "scene_scale": scene_scale,
            "rms_residual": float(np.sqrt((residuals ** 2).mean())),
            "p95_residual": float(np.percentile(np.abs(residuals), 95)),
        },
        "sanity_check": stats,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(plane, f, indent=2)
    print(f"\n[saved] {args.out}")

    plot_diagnostics(xyz, ransac_mask, slab_mask, normal, d, centroid, image_list,
                     "04_plane_proxy", args.show)


if __name__ == "__main__":
    main()
