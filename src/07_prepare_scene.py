"""
Stage 7 — Prepare the dense-pipeline scene bundle.

Builds a `scene/` directory laid out the way COLMAP's dense stage expects
(`images/` + `sparse/0/`) from either of two sources:

  --zip PATH        garuda_colmap_data.zip-style archive: images + sparse/0
                     bundled together. This is the one file that needs to
                     move Mac -> ROG (927 MB, already built).
  --images/--sparse  build directly from local folders instead of a zip
                     (useful for testing on the Mac, where images/ + the
                     colmap_work/sparse/0 model already exist).

Unlike the (abandoned) 3D Gaussian Splatting plan, this does NOT downscale
images -- dense MVS and texture quality both benefit from full resolution,
and COLMAP caps its own *working* resolution internally via
--PatchMatchStereo.max_image_size (see 09_dense_stereo.py), not via
pre-shrunk files on disk.

Note: colmap_work/images/ is 99 symlinks into ../../images/{low,mid,high}/.
Symlinks don't survive a zip/transfer to another machine, so --images should
point at the real images/ folder, not colmap_work/images/, when building a
zip to transfer. Reading directly from an already-unzipped scene/ (which has
real files) is unaffected.

After building, the scene is validated against the known-good reconstruction
stats (97 registered images, 64,945 points, see colmap_work/sparse/0) so a
mismatch is caught here rather than silently producing a bad dense
reconstruction downstream.

Output: <scene>/images/, <scene>/sparse/0/, output/07_prepare_scene.json
"""

import argparse
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dense_common import REPO_ROOT, DEFAULT_SCENE, write_json
from colmap_model import load_model

EXPECTED_IMAGES = 97   # registered poses in colmap_work/sparse/0
EXPECTED_POINTS = 64945
COUNT_TOLERANCE = 0.05  # warn (not fail) if outside +/-5%


def extract_zip(zip_path, scene_dir):
    print(f"Extracting {zip_path} -> {scene_dir}")
    os.makedirs(scene_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        # garuda_colmap_data.zip has images/*.JPG and sparse/0/* at top level
        # (or nested one directory deep) -- handle both.
        top_dirs = {n.split("/")[0] for n in names if "/" in n}
        z.extractall(scene_dir)

    # If everything landed under a single extra top-level folder, flatten it.
    entries = [e for e in os.listdir(scene_dir) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(scene_dir, entries[0])):
        nested = os.path.join(scene_dir, entries[0])
        for item in os.listdir(nested):
            shutil.move(os.path.join(nested, item), os.path.join(scene_dir, item))
        os.rmdir(nested)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _find_images_recursive(images_dir):
    """
    Walk images_dir and return {basename: full_path} for every image file,
    however deep. The real images/ folder here is split into high/low/mid
    subfolders, but COLMAP's sparse model (colmap_work/sparse/0) was built
    against a flat namespace (colmap_work/images/ symlinks all sit directly
    in one folder) -- image names in images.txt have no subfolder prefix, so
    the dense workspace must be flattened the same way or image_undistorter
    won't find matches for the registered names.
    """
    found = {}
    for root, _dirs, files in os.walk(images_dir):
        for fname in files:
            if fname.endswith(IMAGE_EXTS):
                if fname in found:
                    print(f"WARNING: duplicate image name {fname!r} "
                          f"({found[fname]} vs {os.path.join(root, fname)}) -- keeping the first one")
                    continue
                found[fname] = os.path.join(root, fname)
    return found


def build_from_folders(images_dir, sparse_dir, scene_dir, copy_images):
    os.makedirs(scene_dir, exist_ok=True)
    dst_images = os.path.join(scene_dir, "images")
    dst_sparse = os.path.join(scene_dir, "sparse", "0")

    print(f"Flattening images: {images_dir} -> {dst_images}")
    if os.path.exists(dst_images):
        shutil.rmtree(dst_images)
    os.makedirs(dst_images, exist_ok=True)

    sources = _find_images_recursive(images_dir)
    if not sources:
        raise RuntimeError(f"No images found under {images_dir} (looked recursively).")

    for fname, src_path in sources.items():
        dst_path = os.path.join(dst_images, fname)
        real_src = os.path.realpath(src_path)  # resolve symlinks either way
        if copy_images:
            shutil.copy2(real_src, dst_path)
        else:
            os.symlink(real_src, dst_path)
    print(f"  {len(sources)} images -> {dst_images}")

    print(f"Copying sparse model: {sparse_dir} -> {dst_sparse}")
    if os.path.exists(dst_sparse):
        shutil.rmtree(dst_sparse)
    shutil.copytree(sparse_dir, dst_sparse)


def validate_scene(scene_dir):
    images_dir = os.path.join(scene_dir, "images")
    sparse_dir = os.path.join(scene_dir, "sparse", "0")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Expected {images_dir} after preparation, not found.")
    if not os.path.isdir(sparse_dir):
        raise FileNotFoundError(f"Expected {sparse_dir} after preparation, not found.")

    n_images_on_disk = len([f for f in os.listdir(images_dir) if f.endswith(IMAGE_EXTS)])

    # Prefer the text model if present (pure-python, no extra dependency);
    # fall back gracefully if only .bin files were copied.
    txt_present = os.path.isfile(os.path.join(sparse_dir, "cameras.txt")) and \
        os.path.isfile(os.path.join(sparse_dir, "images.txt")) and \
        os.path.isfile(os.path.join(sparse_dir, "points3D.txt"))

    result = {
        "images_on_disk": n_images_on_disk,
        "sparse_text_present": txt_present,
        "n_registered_images": None,
        "n_points": None,
        "warnings": [],
    }

    if txt_present:
        cameras, images, points3D = load_model(sparse_dir)
        n_points = points3D["xyz"].shape[0]  # points3D is a dict of arrays, not a list
        result["n_registered_images"] = len(images)
        result["n_points"] = n_points

        if abs(len(images) - EXPECTED_IMAGES) / EXPECTED_IMAGES > COUNT_TOLERANCE:
            result["warnings"].append(
                f"Registered image count {len(images)} differs from the known-good "
                f"colmap_work/sparse/0 count ({EXPECTED_IMAGES}) by more than "
                f"{COUNT_TOLERANCE:.0%} -- check this is the right sparse model."
            )
        if abs(n_points - EXPECTED_POINTS) / EXPECTED_POINTS > COUNT_TOLERANCE:
            result["warnings"].append(
                f"Point count {n_points} differs from the known-good model "
                f"({EXPECTED_POINTS}) by more than {COUNT_TOLERANCE:.0%}."
            )
    else:
        result["warnings"].append(
            "sparse/0 has no cameras.txt/images.txt/points3D.txt (only .bin?) -- "
            "skipping the registered-image/point-count sanity check. COLMAP's own "
            "tools will still read the .bin files fine."
        )

    if n_images_on_disk == 0:
        raise RuntimeError(f"No images found in {images_dir} after preparation.")
    for w in result["warnings"]:
        print(f"WARNING: {w}")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default=None, help="garuda_colmap_data.zip-style archive (images/ + sparse/0/)")
    parser.add_argument("--images", default=None, help="Existing images folder (used with --sparse instead of --zip)")
    parser.add_argument("--sparse", default=None, help="Existing sparse/0 model folder (used with --images)")
    parser.add_argument("--out", default=DEFAULT_SCENE, help=f"Output scene directory (default: {DEFAULT_SCENE})")
    parser.add_argument("--symlink-images", action="store_true",
                         help="Symlink instead of copy when building from --images/--sparse "
                              "(fine for same-machine testing; do NOT use before zipping for transfer)")
    args = parser.parse_args()

    if args.zip:
        extract_zip(args.zip, args.out)
    elif args.images and args.sparse:
        build_from_folders(args.images, args.sparse, args.out, copy_images=not args.symlink_images)
    else:
        default_zip = os.path.join(REPO_ROOT, "garuda_colmap_data.zip")
        default_images = os.path.join(REPO_ROOT, "images")
        default_sparse = os.path.join(REPO_ROOT, "colmap_work", "sparse", "0")
        if os.path.isfile(default_zip):
            print(f"No --zip/--images given; using {default_zip}")
            extract_zip(default_zip, args.out)
        elif os.path.isdir(default_images) and os.path.isdir(default_sparse):
            print(f"No --zip given; building from {default_images} + {default_sparse}")
            build_from_folders(default_images, default_sparse, args.out, copy_images=not args.symlink_images)
        else:
            parser.error(
                "Pass --zip PATH, or --images PATH --sparse PATH. "
                f"(Also checked for {default_zip} and {default_images}+{default_sparse}, neither found.)"
            )

    result = validate_scene(args.out)
    result["scene_dir"] = args.out
    write_json("07_prepare_scene", result)

    print(f"\nScene ready at {args.out}")
    print(f"  images on disk:      {result['images_on_disk']}")
    print(f"  registered images:   {result['n_registered_images']}")
    print(f"  points:              {result['n_points']}")
    if not result["warnings"]:
        print("  sanity check:        OK (matches known-good reconstruction)")


if __name__ == "__main__":
    main()
