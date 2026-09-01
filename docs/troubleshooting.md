# Troubleshooting

Organized by stage. If a stage's own error message already points here with
a specific anchor, jump to that section; otherwise search for the symptom.

## Stage 7 — prepare scene

**"No images found under ... (looked recursively)"** — check `--images`
points at the folder *containing* `high/low/mid` (or however your images
are organized), not a specific subfolder.

**Warning: registered image / point count differs from the known-good
model by more than 5%** — not a crash, but worth reading before continuing.
Usually means `--sparse` points at the wrong model directory (e.g. an older
or partial reconstruction). The known-good numbers are 97 registered images,
64,945 points, from `colmap_work/sparse/0/`.

**Duplicate image name warning** — two different subfolders contain a file
with the same name; the first one found wins and the rest are skipped. Only
matters if that's not the file you expected — rename or reorganize source
folders if so.

## Stage 8 — undistort

**`FileNotFoundError: Could not find 'colmap' on PATH`** — see
[setup_rog.md](setup_rog.md#colmap) (or `setup_mac.md` if running here).
Either add `colmap`/`colmap.exe` to `PATH`, set `COLMAP_BIN`, or pass
`--colmap-bin` directly.

**Missing-DLL error running `colmap.exe`** — Windows only. Install the
Visual C++ Redistributable (x64); this is not a sign the CUDA build itself
is broken.

## Stage 9 — dense stereo (the VRAM-relevant one)

**Automatic retry already handles a single OOM** — `09_dense_stereo.py`
halves `--max-image-size` and retries up to `--max-retries` times (default
3) down to `--floor` (default 800). You'll see this in the output:

```
Detected a CUDA out-of-memory failure. Retrying at --PatchMatchStereo.max_image_size 1200...
```

**Still OOM at the floor** — the escalation ladder, in order:

1. Lower `--floor` further (e.g. 600).
2. Add `--cache-size 8` (or lower) to shrink COLMAP's host-side image
   cache, which indirectly reduces GPU transfer pressure.
3. Confirm nothing else is holding VRAM (close other GPU applications,
   check `nvidia-smi` before starting).

**Fails immediately, not an OOM message** — the retry loop only fires on a
recognized out-of-memory pattern (`out of memory`,
`CUDA_ERROR_OUT_OF_MEMORY`, `std::bad_alloc`, etc. — see `OOM_MARKERS` in
`src/dense_common.py`). Any other failure stops immediately and prints the
stderr tail; read that first — common causes are a workspace from a skipped
Stage 8, or `--gpu-index` not matching an actual device (`nvidia-smi -L` to
list GPU indices).

**It's just slow** — expected. Dense stereo across 97 images, even capped
at 2400px, takes real time. This is the workload the 1650's CUDA cores are
suited to (unlike the sparse stage's 56-minute Mac CPU SIFT run) — let it
run rather than assuming it's stuck. `patch_match_stereo` prints per-image
progress; if that's advancing, it's working.

## Stage 10 — fuse + mesh

**`fused_points` much lower than expected / near-zero** — check Stage 9
actually completed (a partial/interrupted `patch_match_stereo` run can leave
a `stereo/` directory that `stereo_fusion` will happily "succeed" against
with almost nothing fused). Re-run Stage 9 if so.

**Hole or noisy region above the eaves** — expected, not a bug. All three
elevation bands are eye-level-and-up walkarounds; nothing looked straight
down onto the top of the relief. Report it as a capture-coverage limitation
in the paper.

**`sparse_points` shows as `null` in the output JSON** — Stage 10 only
computes the sparse-vs-dense comparison if `<sparse>/points3D.txt` exists
(the text-format model). If only `.bin` files were copied during Stage 7,
this comparison is skipped gracefully — the fusion/mesh stats themselves
are unaffected. Pass `--sparse` explicitly if the model lives somewhere
non-default.

## Stage 11 — OpenMVS

This stage is marked best-effort in `src/11_texture_openmvs.py` and
[dense_pipeline.md](dense_pipeline.md#stage-11-optional-stretch--real-uv-texture-atlas-via-openmvs)
— it was not executable-tested against a live OpenMVS install, unlike every
other stage in this pipeline.

**Command fails with an unrecognized-flag error** — OpenMVS's CLI has
changed across releases. Run `InterfaceCOLMAP --help` and
`TextureMesh --help` on your installed version and compare against the
flags in `src/11_texture_openmvs.py`; adjust the script if they've moved.

**Not worth debugging under time pressure** — this stage is optional. The
Stage 10 vertex-colour mesh is a complete, real, textured (per-vertex)
mesh on its own. Skip Stage 11 and go straight to Stage 12/13 if OpenMVS
is fighting you.

## Stage 12 — export

**`ERROR: trimesh is not installed`** — `pip install -r requirements.txt`.

**Exported `.glb` looks wrong / geometry missing when opened elsewhere** —
open the *source* `.ply`/`.obj` in Meshlab/Blender first to isolate whether
the problem is upstream (Stage 10/11's output) or in the trimesh conversion
itself.

## Stage 13 — turntable render

**`[Open3D Error] EGL Headless is not supported on this platform`** — this
is why `13_turntable_render.py` uses the legacy
`Visualizer(visible=False)` API rather than `OffscreenRenderer`. If you see
this error, something reintroduced `OffscreenRenderer` — check for a stray
edit. See [setup_mac.md](setup_mac.md#a-real-gotcha-this-pipeline-already-hit).

**`ERROR: Open3D could not create an off-screen render window`** — rare;
seen on machines with no GPU driver / display subsystem available at all
(e.g. some CI containers). Fallback: open the `.glb`/`.obj` in Meshlab or
Blender and record a turntable manually — the mesh itself is unaffected,
this is purely a rendering-tool limitation on that specific machine.

**Video plays but looks static** — check `--frames` isn't set to 1, and
that `view_ctl.rotate(...)` is actually being called each iteration (it is,
in the shipped script — this note is here because it was the first thing
checked when validating the render actually rotates, via frame-to-frame
pixel differencing, before trusting the output).

## Orchestrator (`run_dense_pipeline.py`)

**A stage fails partway through** — the orchestrator prints which stage
failed and the exact resume command, e.g.:

```
Stage 'undistort' failed (exit 1). Stopping.
Fix and re-run with --resume --from undistort to continue from here.
```

Fix the underlying issue (see the stage-specific section above), then
re-run with `--resume --from <stage>` rather than starting over — `--resume`
skips any stage whose expected output already exists.

**Want to see what would run before committing to it** — `--dry-run` prints
every command in order (including binary-not-found errors) without running
anything.
