# Local 3DGS viewer

A single-page WebGL viewer for the trained splat, for when
`supersplat.playcanvas.com` won't open or won't load the file.

## Use

Opening `index.html` straight from disk (`file://`) is blocked by the
browser for ES modules — serve it instead:

```bash
cd cv-project-codes
python -m http.server 8000
```

Then open <http://localhost:8000/viewer/> and drag in
`exports/splat.ply` (or a `.splat` / `.ksplat`).

Controls: drag = orbit, scroll = zoom, right-drag = pan.

## Notes

- Loads the standard PLY that `ns-export gaussian-splat` writes — no
  conversion needed.
- Needs internet the first time: it pulls `three` and
  `@mkkellogg/gaussian-splats-3d` from jsDelivr. After that the browser
  caches them.
- A 200-frame scene can be a 200–500 MB PLY; give it a moment. If it's
  sluggish, open the PLY once in SuperSplat (or `ns-export` with a lower
  Gaussian count) and save a compressed `.splat`, then view that here.
- If the model appears upside-down, that's the camera-up convention —
  just orbit around; the geometry is fine.
