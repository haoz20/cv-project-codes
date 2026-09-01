"""
Resize a folder of images by a scale factor (e.g. 0.5 = half resolution).

Usage (run on the Windows laptop, same folder structure as the pipeline):
    python resize_images.py --input images --output images_small --scale 0.5

After running, regenerate transforms.json pointed at the output folder so the
camera intrinsics (fl_x, fl_y, cx, cy, w, h) get recomputed correctly for the
new resolution instead of hand-edited:
    python colmap2nerf.py --text sparse/0 --images images_small --out transforms.json
"""

import argparse
import os
import sys

import cv2


def resize_folder(input_dir, output_dir, scale):
    if not os.path.isdir(input_dir):
        print(f"ERROR: input folder not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(exts))

    if not files:
        print(f"ERROR: no images found in {input_dir}")
        sys.exit(1)

    print(f"Resizing {len(files)} images from {input_dir} -> {output_dir} (scale={scale})")

    for i, fname in enumerate(files, 1):
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)

        img = cv2.imread(src_path)
        if img is None:
            print(f"  WARNING: could not read {fname}, skipping")
            continue

        h, w = img.shape[:2]
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(dst_path, resized)

        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {fname}: {w}x{h} -> {new_w}x{new_h}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="images", help="Input folder (default: images)")
    parser.add_argument("--output", default="images_small", help="Output folder (default: images_small)")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor, e.g. 0.5 = half resolution (default: 0.5)")
    args = parser.parse_args()

    resize_folder(args.input, args.output, args.scale)


if __name__ == "__main__":
    main()
