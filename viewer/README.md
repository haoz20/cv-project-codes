# Local 3DGS viewer

A React + Vite viewer for comparing two trained splats side by side
(e.g. closed-lid vs. open-lid), for when `supersplat.playcanvas.com`
won't open or won't load the file.

## Use

```bash
cd viewer
npm install
npm run dev
```

Then open the printed `http://localhost:5173/` URL. Each pane loads
its own file independently — drag in `exports/splat.ply` (or a
`.splat` / `.ksplat`) on the left, and a second export on the right.

Controls: drag = orbit, scroll = zoom, right-drag = pan.

- **Left panel**: background color picker (applies to both views).
- **Right panel**: auto-orbit toggle + speed (drives both cameras in
  sync) and a reset-cameras button.

`npm run build` produces a static `dist/` bundle if you want to host
it (e.g. behind `python -m http.server` from inside `dist/`).

## Notes

- Loads the standard PLY that `ns-export gaussian-splat` writes — no
  conversion needed.
- A 200-frame scene can be a 200–500 MB PLY; give it a moment. If it's
  sluggish, open the PLY once in SuperSplat (or `ns-export` with a lower
  Gaussian count) and save a compressed `.splat`, then view that here.
- If the model appears upside-down, that's the camera-up convention —
  just orbit around; the geometry is fine.
