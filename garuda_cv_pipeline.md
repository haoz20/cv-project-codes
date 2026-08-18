# CV Pipeline — Garuda 3D Rendering Project (CSX4213)

**Subject:** Garuda statue, Garuda Museum, Samut Prakan
**Approach:** Classical mesh-based photogrammetry (feature matching → SfM → MVS → surface reconstruction → texture mapping → rendering)
**Team:** Sam Yati, Thiri Htet, Swan Htet Aung — Vincent Mary School of Engineering, Science and Technology, Assumption University

---

## 1. Image Acquisition

- Capture one Garuda statue only (prefer the large freestanding sculpture over flat/low-relief emblems — more volumetric detail for reconstruction).
- Circular multi-view sweep around the subject.
- 60–80% overlap between consecutive photos.
- 2–3 height levels (low, eye-level, above) for full coverage.
- Consistent focal length throughout the shoot.
- Consistent lighting — avoid shooting across changing light/shadow conditions.
- Target: 50–100+ photos.
- Optional: a few close-up shots of fine detail areas (face, wings), kept consistent with the main sweep.

## 2. Feature Detection & Matching

- Detect distinctive keypoints in each photo.
  - **SIFT** — standard choice, more accurate.
  - **ORB** — faster, less accurate.
- Match corresponding keypoints across image pairs (e.g., `cv2.BFMatcher` or FLANN).
- Tooling: OpenCV (`cv2.SIFT_create()`), same approach as Ch. 8 coursework.

## 3. Camera Pose Estimation

- Estimate the essential/fundamental matrix between matched image pairs.
- Recover each camera's relative rotation and translation.

## 4. Structure from Motion (SfM)

- Incrementally or globally register all cameras into one shared coordinate frame.
- Triangulate matched points into a **sparse 3D point cloud**.
- Refine via bundle adjustment.

## 5. Multi-View Stereo (MVS)

- Using known camera poses, densify the sparse cloud by matching pixels (not just keypoints) across many overlapping views.
- Output: **dense 3D point cloud**.

## 6. Surface Reconstruction

- Convert the dense point cloud into a connected polygonal mesh.
- **Poisson surface reconstruction** — standard, robust to noise and missing data.

## 7. Texture Mapping

- Project color from the original photos back onto the mesh faces.
- Blend seams across source images taken under varying lighting.
- Output: textured mesh reflecting the statue's true appearance, not just geometry.

## 8. 3D Rendering / Export

- Export the textured mesh (`.obj` / `.ply` / `.glb`).
- View, rotate, and render from any angle in a 3D viewer.

---

## Tooling Plan (MacBook Air M2, 16GB)

| Stage(s) | Tool | Notes |
|---|---|---|
| 2 (feature detection/matching) | OpenCV (Python, Anaconda) | Own code — demonstrates the stage directly, reuses Ch. 8 approach |
| 3–7 (pose → SfM → MVS → mesh → texture) | **COLMAP** | Free, open source. Install via `brew install colmap`. No CUDA on M2 → MVS runs on CPU (slower, but workable at this photo count). GUI + CLI. |
| 3–7 (alternative) | Meshroom (AliceVision) | Free, GUI-based, same stages. Also loses GPU acceleration on Apple Silicon. |
| 3–7 (fallback if free tools struggle) | Agisoft Metashape | Commercial (free trial / student pricing). Native macOS app, fastest/most polished results. |

**Recommended workflow:** Use OpenCV for Stage 2 to write and explain your own feature-matching code for the report, then run the full photo set through COLMAP for Stages 3–7 to produce the final textured mesh. Screenshot/export COLMAP's output for the Results section.

---

## Open Questions / To Confirm

- [ ] Confirm exact photo count captured at the museum visit
- [ ] Confirm which Garuda statue/emblem was used as the subject
- [ ] Note actual processing time and any failure points (holes in mesh, texture seams, etc.) for the Results/Analysis section
- [ ] Decide final export format for submission (.obj / .ply / .glb)
