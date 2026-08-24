# Progress Check — Speaking Script (~3 min, single speaker)

Cues in `[brackets]` mark when to switch slides / show a figure. Figures referenced are in
`output/` — generate them fresh before presenting with:

```bash
/opt/anaconda3/bin/python src/01_detect_features.py
/opt/anaconda3/bin/python src/02_match_pairs.py
/opt/anaconda3/bin/python src/03_verify_matches.py
```

---

[SLIDE: Title — Garuda 3D Reconstruction]

Good [morning/afternoon]. Our project reconstructs a 3D digital model of a Garuda statue from
the Garuda Museum in Samut Prakan — Thailand's largest collection of carved royal Garuda
emblems. Right now these pieces only exist to the public as static photographs, so you can't
rotate one to see the wing detail or get a sense of its actual depth. Our goal is to turn a set
of ordinary photos, taken with a personal camera on a single museum visit, into a textured 3D
mesh you can view from any angle.

[SLIDE: Pipeline overview]

We're using a classical photogrammetry pipeline, in eight stages. It starts with image
acquisition — a circular sweep of photos around the statue with heavy overlap. Then feature
detection and matching, finding the same points across different photos. From those matches we
estimate each camera's pose, then run Structure from Motion to build a sparse 3D point cloud,
and Multi-View Stereo to densify it. Finally, Poisson surface reconstruction turns the point
cloud into a mesh, texture mapping projects the original photo colors back onto it, and we
export a renderable 3D model. Feature matching we're implementing ourselves in OpenCV; the
camera pose through mesh stages will run through COLMAP.

[SLIDE: Stage 2 demo — keypoints]

This week we have working code for stage two — feature detection and matching — which is the
piece that everything downstream depends on. We compare two detectors: SIFT, which is slower but
more accurate, and ORB, which is faster. [SHOW FIGURE: 01_keypoints_sift.png] On our test images
SIFT finds around twenty-eight hundred keypoints in about a quarter of a second, concentrated on
corners and textured surfaces — exactly where you'd want them for a statue with fine carved
detail.

[SLIDE: Stage 2 demo — matches]

[SHOW FIGURE: 02_matches_sift.png] For matching, we use Lowe's ratio test to keep only reliable
correspondences between image pairs — comparing the best match against the second-best and
discarding anything ambiguous. We also swept the ratio threshold from 0.6 to 0.8 to justify our
0.75 cutoff, rather than just guessing a number. On this pair, SIFT gives us six hundred good
matches versus ORB's four hundred twenty — SIFT wins on quality, ORB wins on speed.

[SHOW FIGURE: 03_inlier_matches.png] We also added a RANSAC geometric check on top, which filters
out matches that aren't physically consistent between the two viewpoints — that gets us to a
sixty-seven percent inlier rate, which is our real match-quality number.

[SLIDE: Next steps]

We haven't done the museum shoot yet, so these results are from placeholder test photos — we'll
plug the real Garuda photos straight into the same code once we have them. Before that, we're
doing a dry-run capture at home to validate our shooting protocol: sixty to eighty percent
overlap, consistent lighting, fixed focal length. Next, we take the verified matches into camera
pose estimation and Structure from Motion through COLMAP.

Happy to take any questions.

---

## Anticipated Q&A

**"Why not just use COLMAP for everything, including feature matching?"**
COLMAP does its own matching internally, but writing our own stage-2 code lets us directly
demonstrate and explain the classical feature-matching concepts from the course — SIFT, ratio
testing, RANSAC — rather than treating it as a black box. We hand off to COLMAP once the pipeline
moves past matching.

**"Why SIFT over ORB if SIFT is slower?"**
Accuracy matters more than speed here — we're processing 50-100 photos once, not in real time,
so the extra runtime cost is worth the better match quality it gives every downstream stage.

**"What happens if the museum photos don't work well?"**
That's exactly why we're doing a dry-run capture first — it validates the overlap and lighting
protocol on a low-stakes subject before we commit to a single museum visit.
