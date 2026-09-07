"""
Stage 01 -- video -> frames.

Decodes the capture video with PyAV (in-process, no ffmpeg subprocess, and
unlike cv2.VideoCapture it reliably handles the HEVC that phones write into
.MOV / .mp4 containers), samples frames at a fixed rate, and writes:

  frames/00001.jpg, 00002.jpg, ...   full-resolution JPEGs
  frames/sharpness.csv               filename, laplacian_var, rank_worst_first

The video is auto-detected as data/kratib.<mov|mp4|...>, or the single
video file in data/, or pass --video explicitly.

sharpness.csv is NOT a filter -- nothing is dropped here. It is a
worst-first reading order for the manual pass that comes next: sort by
rank_worst_first, look at the low-scoring frames, delete the ones that are
actually motion-blurred from frames/, then run Stage 02. Blurred frames do
not just get ignored by COLMAP -- they produce bad matches that drag down
pose accuracy for their neighbours too.

Usage:
    python src/01_extract_frames.py                       # auto-find data/kratib.*
    python src/01_extract_frames.py --video data/kratib.MOV --fps 3
    python src/01_extract_frames.py --rotate 90           # portrait .MOV coming out sideways
    python src/01_extract_frames.py --dry-run
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

from common import FRAMES_DIR, find_capture_video


def laplacian_var(bgr):
    """Variance of the Laplacian -- standard focus metric. Higher = sharper."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def apply_rotation(bgr, deg):
    if deg == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def has_rotation_metadata(stream):
    """
    True if the stream carries display-rotation info. PyAV does NOT apply it
    to decoded frames, so a portrait .MOV would extract sideways -- we warn
    and let the user correct it with --rotate rather than guess the sign.
    """
    try:
        if int(stream.metadata.get("rotate", 0)) % 360:
            return True
    except (TypeError, ValueError):
        pass
    try:
        for sd in stream.side_data or []:
            if getattr(sd.type, "name", "") == "DISPLAYMATRIX":
                return True
    except Exception:
        pass
    return False


def extract(video_path, out_dir, fps, jpeg_quality, rotate):
    import av  # local import: keeps --help / --dry-run working without PyAV

    if not os.path.isfile(video_path):
        sys.exit(f"ERROR: video not found: {video_path}")

    print(f"Opening {video_path}")
    container = av.open(video_path)
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate) if stream.average_rate else 30.0
    duration = float(container.duration / 1_000_000) if container.duration else None
    msg = f"  source: {stream.width}x{stream.height} @ {src_fps:.3f} fps"
    if duration:
        msg += f", ~{duration:.1f}s"
    print(msg)
    print(f"  sampling at {fps} fps (>= {1.0 / fps:.2f}s between kept frames)")

    if rotate:
        print(f"  rotating every frame {rotate} deg clockwise (--rotate)")
    elif has_rotation_metadata(stream):
        print("  NOTE: this video carries rotation metadata, which PyAV does not apply.")
        print("        If the extracted frames come out sideways, delete frames/ and")
        print("        re-run with  --rotate 90  (or 180 / 270).")

    os.makedirs(out_dir, exist_ok=True)
    interval = 1.0 / fps
    next_t = 0.0
    kept = 0
    rows = []
    for idx, frame in enumerate(container.decode(video=0)):
        if frame.pts is None:
            t = idx / src_fps
        else:
            t = float(frame.pts * stream.time_base)
        if t < next_t:
            continue
        next_t = t + interval

        bgr = frame.to_ndarray(format="bgr24")
        if rotate:
            bgr = apply_rotation(bgr, rotate)
        kept += 1
        name = f"{kept:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        rows.append((name, laplacian_var(bgr)))
        if kept % 20 == 0:
            print(f"  {kept} frames...")

    container.close()

    if not rows:
        sys.exit("ERROR: decoded 0 frames -- is the video readable / not empty?")

    # rank 1 = blurriest, so the manual pass can read the list top-down.
    order = sorted(range(len(rows)), key=lambda i: rows[i][1])
    rank = {i: r + 1 for r, i in enumerate(order)}
    csv_path = os.path.join(out_dir, "sharpness.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "laplacian_var", "rank_worst_first"])
        for i, (name, var) in enumerate(rows):
            w.writerow([name, f"{var:.2f}", rank[i]])

    vals = np.array([v for _, v in rows])
    print(f"\n[saved] {kept} frames -> {out_dir}")
    print(f"[saved] {csv_path}")
    print(f"  sharpness  min {vals.min():.0f}   median {np.median(vals):.0f}   max {vals.max():.0f}")
    print(f"\nNext: sort {os.path.basename(csv_path)} by rank_worst_first, delete blurred")
    print(f"      frames from {out_dir}/, then run  python src/02_process_data.py")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=None,
                        help="Input video (default: auto-detect data/kratib.* or the "
                             "single video in data/)")
    parser.add_argument("--out", default=FRAMES_DIR,
                        help="Output frames folder (default: frames/)")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frames per second to sample (default: 2)")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="Rotate frames clockwise (for portrait .MOV that decodes sideways)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    video = args.video or find_capture_video()

    if args.dry_run:
        shown = video or "data/kratib.<mov|mp4>  (none found yet)"
        print(f"[dry-run] PyAV decode {shown} @ {args.fps} fps "
              f"-> {args.out}/*.jpg + sharpness.csv")
        return

    if video is None:
        sys.exit("ERROR: no video found. Put your capture at data/kratib.MOV (or .mp4), "
                 "or pass --video PATH.")
    extract(video, args.out, args.fps, args.jpeg_quality, args.rotate)


if __name__ == "__main__":
    main()
