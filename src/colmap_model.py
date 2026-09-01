"""
Minimal reader for a COLMAP sparse model (text export).

Used by 04_plane_proxy.py and 05_view_dependent_render.py. Both need camera
intrinsics, image poses and the 3D point cloud, so the parsing lives here
rather than being duplicated — the numeric-prefixed scripts can't import
each other (see the note at the top of 03_verify_matches.py).

Only the text format is read. `colmap model_converter` already wrote
cameras.txt / images.txt / points3D.txt alongside the .bin files in
colmap_work/sparse/0, and the text format needs no struct unpacking.

Pose convention (COLMAP's): a world point X maps into camera coordinates as
    X_cam = R @ X_world + t
with R = qvec2rotmat(qvec) and t = tvec. The camera centre in world
coordinates is therefore C = -R.T @ t.
"""

import os

import numpy as np


class Camera:
    """One COLMAP camera model (intrinsics), shared by many images."""

    def __init__(self, camera_id, model, width, height, params):
        self.id = camera_id
        self.model = model
        self.width = width
        self.height = height
        self.params = np.asarray(params, dtype=np.float64)

    def K(self, scale=1.0):
        """
        3x3 pinhole intrinsic matrix, optionally for a downscaled image.

        `scale` is the factor the image was resized by (0.2 for a 5x
        downscale), applied to focal length and principal point alike.
        """
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            f, cx, cy = self.params[0], self.params[1], self.params[2]
            fx = fy = f
        elif self.model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
            fx, fy, cx, cy = self.params[0], self.params[1], self.params[2], self.params[3]
        else:
            raise NotImplementedError(f"Unsupported camera model: {self.model}")

        return np.array(
            [[fx * scale, 0.0, cx * scale],
             [0.0, fy * scale, cy * scale],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def dist_coeffs(self):
        """
        Radial/tangential coefficients in OpenCV's (k1, k2, p1, p2) order.

        Our capture used SIMPLE_RADIAL (a single k1); the others are filled
        with zeros so cv2.undistort can be called uniformly.
        """
        if self.model == "SIMPLE_PINHOLE":
            return np.zeros(4)
        if self.model == "SIMPLE_RADIAL":
            return np.array([self.params[3], 0.0, 0.0, 0.0])
        if self.model == "RADIAL":
            return np.array([self.params[3], self.params[4], 0.0, 0.0])
        if self.model == "PINHOLE":
            return np.zeros(4)
        if self.model in ("OPENCV", "FULL_OPENCV"):
            return np.array(self.params[4:8], dtype=np.float64)
        raise NotImplementedError(f"Unsupported camera model: {self.model}")


class Image:
    """One registered photo: its pose in the reconstruction plus its filename."""

    def __init__(self, image_id, qvec, tvec, camera_id, name):
        self.id = image_id
        self.qvec = np.asarray(qvec, dtype=np.float64)
        self.tvec = np.asarray(tvec, dtype=np.float64)
        self.camera_id = camera_id
        self.name = name

    @property
    def R(self):
        """Rotation world -> camera."""
        return qvec2rotmat(self.qvec)

    @property
    def C(self):
        """Camera centre in world coordinates."""
        return -self.R.T @ self.tvec

    @property
    def viewing_direction(self):
        """Unit vector along the optical axis, pointing into the scene."""
        return self.R.T @ np.array([0.0, 0.0, 1.0])


def qvec2rotmat(qvec):
    """Convert a COLMAP quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def read_cameras_text(path):
    """Parse cameras.txt -> {camera_id: Camera}."""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=[float(p) for p in parts[4:]],
            )
    return cameras


def read_images_text(path):
    """
    Parse images.txt -> {image_id: Image}.

    Every image occupies two lines: the pose line, then its POINTS2D list.
    The 2D observations aren't needed here (they're what the sparse model was
    built from, not what we render with), so the second line is skipped —
    which also keeps this cheap on a 36 MB images.txt.
    """
    images = {}
    with open(path) as f:
        lines = (ln.strip() for ln in f)
        lines = [ln for ln in lines if ln and not ln.startswith("#")]

    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        images[image_id] = Image(
            image_id=image_id,
            qvec=[float(p) for p in parts[1:5]],
            tvec=[float(p) for p in parts[5:8]],
            camera_id=int(parts[8]),
            name=parts[9],
        )
    return images


def read_points3D_text(path, min_track=0, max_error=None):
    """
    Parse points3D.txt -> (xyz, rgb, error, track_length) arrays.

    Parameters
    ----------
    min_track : int
        Drop points seen by fewer than this many images. Short tracks are the
        least reliable points in the cloud and would drag a plane fit around.
    max_error : float or None
        Drop points whose mean reprojection error exceeds this (px).

    Returns
    -------
    dict with keys 'xyz' (N,3), 'rgb' (N,3), 'error' (N,), 'track' (N,).
    """
    xyz, rgb, error, track = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # TRACK[] is a flat list of (IMAGE_ID, POINT2D_IDX) pairs.
            track_len = (len(parts) - 8) // 2
            err = float(parts[7])
            if track_len < min_track:
                continue
            if max_error is not None and err > max_error:
                continue
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
            error.append(err)
            track.append(track_len)

    return {
        "xyz": np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        "rgb": np.asarray(rgb, dtype=np.uint8).reshape(-1, 3),
        "error": np.asarray(error, dtype=np.float64),
        "track": np.asarray(track, dtype=np.int32),
    }


def load_model(model_dir, min_track=0, max_error=None):
    """
    Read a whole sparse model directory.

    Returns (cameras, images, points) — see the read_* functions above.
    """
    cameras = read_cameras_text(os.path.join(model_dir, "cameras.txt"))
    images = read_images_text(os.path.join(model_dir, "images.txt"))
    points = read_points3D_text(
        os.path.join(model_dir, "points3D.txt"), min_track=min_track, max_error=max_error
    )
    return cameras, images, points
