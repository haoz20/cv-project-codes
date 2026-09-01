# Progress Log — Garuda 3D Reconstruction (CSX4213)

Running log of what's been done, decided, and still open. Newest entries at the top. Append a
new dated section each time something meaningful happens — a shoot, a pipeline run, a decision,
a paper edit — rather than editing old entries (except to correct a mistake). This is the log
to skim before a progress check or before picking up work again after a gap.

Reference docs: [`garuda_cv_pipeline.md`](garuda_cv_pipeline.md) (pipeline design),
[`README.md`](README.md) (how to run the code), [`PRESENTATION_SCRIPT.md`](PRESENTATION_SCRIPT.md).

---

## Quick status

| Stage | Status |
|---|---|
| 1. Image acquisition | ✅ Done — 100 photos, museum visit 2026-08-22 |
| 2. Feature detection & matching | ✅ Code done, validated on real photos |
| 3. Camera pose estimation | ⬜ Not started — via COLMAP |
| 4. Structure from Motion | ⬜ Not started — via COLMAP |
| 5. Multi-View Stereo | ⬜ Not started — via COLMAP |
| 6. Surface reconstruction | ⬜ Not started |
| 7. Texture mapping | ⬜ Not started |
| 8. Export / rendering | ⬜ Not started |
| Paper: Abstract / Intro / Lit Review | ✅ Written (Intro ¶2 has an unresolved contradiction — see Open Items) |
| Paper: Methodology / Results / Conclusion | ⬜ Still placeholder text in the docx |

---

## 2026-08-24 — Real-photo validation of Stage 2, pipeline decisions for the 180° relief capture

**What happened:** Reviewed the `images/` folder from the museum visit and ran the Stage 2 code
(`src/01`–`03`) against real photo pairs instead of the placeholder desk images used in the first
commit.

**Subject confirmed:** A wall/shelf-mounted Garuda emblem (flat-backed relief carving — wings,
arms, legs, headdress in raised detail), not a freestanding statue. This is why capture is
capped at ~180°: the back is physically flush against the wall, there's nothing to shoot there.
Differs from `garuda_cv_pipeline.md`'s original preference for a freestanding sculpture, but
one-sided reliefs are a completely standard photogrammetry case.

**Capture stats (EXIF-verified):** 100 photos total — low: 20, mid: 28, high: 51. iPhone 17 Pro
Max, fixed focal length (6.765mm) throughout, entire shoot within a 19-minute window on
2026-08-22 (consistent lighting). One likely-unusable frame: `images/high/IMG_6108.JPG` (steep
near-ceiling angle, visibly soft/blurry) — flag for culling before any COLMAP run.

**Real Stage 2 numbers:**

| Pair | SIFT keypoints | Good matches (ratio 0.75) | RANSAC inliers | Inlier ratio |
|---|---|---|---|---|
| `mid/IMG_6003` ↔ `IMG_6004` (consecutive, within-band) | 2523 / 3363 | 679 | 570 | 84% |
| `mid/IMG_6015` ↔ `IMG_6016` (face/detail pair — best result, used for demo) | 4178 / 4221 | 2342 | 2028 | **87%** |
| `low/IMG_6050` ↔ `high/IMG_6051` (weakest cross-band boundary, arbitrary pair) | 2029 / 1916 | 14 | **8** | 57% |

Within-band overlap is excellent (84–87% inlier ratio) — the 60–80% overlap target was clearly
hit. The low→high boundary is the fragile spot: 8 RANSAC inliers is the bare minimum
`cv2.findFundamentalMat` needs to fit at all. This was only tested on one arbitrary frame pair
(first/last of each band); COLMAP's exhaustive matcher searches every pair and will likely find a
stronger bridge — but it's the thing to watch when the full reconstruction runs.

**Decisions made:**
1. **Keep the mesh-based SfM → MVS → Poisson → texture pipeline** (not image-based rendering).
   The Abstract, Lit Review §D, and this pipeline doc all already commit to it — see Open Items
   below for the one paragraph in the paper that still contradicts this.
2. **Use Screened Poisson + density-based trimming (or ball-pivoting/alpha-shape) instead of
   plain Poisson** for surface reconstruction. The subject is a one-sided relief with no back
   geometry to scan; naive closed-volume Poisson will fabricate a fake back surface to seal the
   mesh. This is standard practice for bas-relief/facade photogrammetry.
3. **Use COLMAP's `exhaustive_matcher`, not `sequential_matcher`, and flatten all photos into one
   folder before running it.** The low/mid/high bands are separate file-numbering blocks
   (6003–6030, 6031–6050, 6051–6108) with only a thin bridge between them (see table above).
   Sequential matching assumes spatial/numeric order and could miss the cross-band links
   entirely, splitting the reconstruction into 3 disconnected models. At 100 images, exhaustive
   matching is cheap enough to just run.

**Artifacts:** `output/01_keypoints_sift.png`, `02_matches_sift.png`, `03_inlier_matches.png`
(currently hold the IMG_6015/6016 demo pair — regenerate before reusing for a different pair,
see README). `data/demo/` holds a persistent copy of the strongest pair for live demos.

---

## 2026-08-18 — Repo bootstrapped, Stage 2 sample code, pushed to GitHub

**What happened:**
- Created `garuda_cv_pipeline.md` (8-stage pipeline design, tooling plan for M2 MacBook Air).
- Wrote and validated Stage 2 code (`src/common.py`, `01_detect_features.py`,
  `02_match_pairs.py`, `03_verify_matches.py`) — SIFT/ORB detection, `BFMatcher` + Lowe's ratio
  test matching, RANSAC fundamental-matrix verification. Runs on the Anaconda interpreter
  (`/opt/anaconda3/bin/python`, cv2 4.12.0 / matplotlib 3.9.2 / numpy 1.26.4).
- Validated against placeholder desk photos (`chapter8/images/img1-3.jpg`, museum photos not
  yet captured at this point) — SIFT: 2847 keypoints, 600 good matches, 67% inlier ratio.
- Wrote `PRESENTATION_SCRIPT.md` — ~3 min single-speaker script for the progress check.
- Repo created and pushed: `github.com/haoz20/cv-project-codes` (public).

**Reviewed the paper draft** (`Garuda-CV-Project.docx`, OneDrive) — found:
- Abstract, Introduction, Literature Review are complete and well-cited (Lowe, Snavely,
  Schönberger, Furukawa, Kazhdan, Waechter, Mildenhall).
- Methodology, Results and Analysis, Conclusion are still placeholder text ("The .").
- **Unresolved contradiction:** Introduction ¶2 describes an image-based-rendering approach
  (view interpolation + layered depth images, "without requiring explicit polygonal geometry"),
  which conflicts with the Abstract, Lit Review §D, and the pipeline doc's mesh-based approach.
  Likely leftover text from an earlier draft of the proposal. **Still not fixed** as of the
  2026-08-24 entry above.

---

## Open items / risks

- [ ] **Paper contradiction:** rewrite Introduction ¶2 (and the "image-based rendering" keyword)
      to match the mesh pipeline described everywhere else in the paper.
- [ ] **Methodology / Results / Conclusion sections are empty** — draft using the real numbers
      and figures in the 2026-08-24 entry above.
- [ ] **Install COLMAP** (`brew install colmap`) — not yet done on this machine.
- [ ] **Cull `images/high/IMG_6108.JPG`** (blurry) before any COLMAP run.
- [ ] **Check oblique-angle coverage** across the full 100-photo set — the relief is fairly flat,
      and a sweep that stays close to fronto-parallel gives SfM little real depth parallax to
      triangulate (risk of a near-planar/homography-degenerate solution). Spot-check angles
      before the full reconstruction run; a few more grazing-angle shots may be worth adding.
- [ ] **Run full COLMAP reconstruction** (exhaustive matching, one flattened folder) and confirm
      it registers all three bands into a single connected sparse model rather than splitting at
      the low/mid or mid/high seams.
- [ ] **Confirm final export format** (.obj / .ply / .glb) for submission.
