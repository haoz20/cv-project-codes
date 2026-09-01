# Garuda dense-pipeline docs

Classical mesh photogrammetry (COLMAP dense MVS → Poisson → texture → export),
picking up from the sparse SfM model already in `colmap_work/sparse/0/`
(97/99 images registered, 64,945 points). See the plan this implements:
`/Users/znhao/.claude/plans/forget-the-whole-current-ethereal-squid.md`.

- **[dense_pipeline.md](dense_pipeline.md)** — the pipeline itself: every
  stage, what it does, exact commands, flags, expected output. Start here.
- **[setup_rog.md](setup_rog.md)** — one-time setup on the ROG (Windows,
  GTX 1650): COLMAP's prebuilt CUDA binary, optionally OpenMVS.
- **[setup_mac.md](setup_mac.md)** — one-time setup on the Mac: Python
  environment, which stages run here.
- **[troubleshooting.md](troubleshooting.md)** — known failure modes and
  fixes, organized by stage.

## Two machines, one pipeline

| Stage | Script | Needs | Runs on |
|---|---|---|---|
| 7. Prepare scene | `src/07_prepare_scene.py` | Python only | either |
| 8. Undistort | `src/08_undistort.py` | COLMAP (CPU) | either |
| 9. Dense stereo | `src/09_dense_stereo.py` | COLMAP + CUDA | **ROG** |
| 10. Fuse + mesh | `src/10_fuse_mesh.py` | COLMAP (light) | either |
| 11. Texture *(optional)* | `src/11_texture_openmvs.py` | OpenMVS | ROG (preferred) |
| 12. Export | `src/12_export_mesh.py` | Python (trimesh) | either — light |
| 13. Turntable render | `src/13_turntable_render.py` | Python (open3d) | either — light |

Run the whole chain with `run_dense_pipeline.py` (see
[dense_pipeline.md](dense_pipeline.md#orchestrator)), or run any stage
individually while tuning — every stage script also works stand-alone.
