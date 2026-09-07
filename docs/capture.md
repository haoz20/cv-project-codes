# Capturing the kratib video

One continuous handheld video. The whole pipeline's quality is set here --
COLMAP solves for **camera** motion, so the object must not move.

## Setup

- **Kratib static** on a **patterned surface** -- a printed cloth, a
  newspaper, a textured placemat. Never a bare white table: a plain
  surface gives COLMAP nothing to match against near the object.
- **Leave the background visible and unchanged.** Do not drape a cloth
  behind it. The room behind the object supplies the stable far features
  that anchor the reconstruction -- woven bamboo alone is repetitive
  texture and matches poorly.
- Bright, **diffuse** light (near a window, or two soft lamps). No hard
  shadows, no flash, no moving light.

## Camera

- **Lock exposure, white balance, and focus** before you start (on iPhone:
  tap-and-hold to get the AE/AF-L lock). Auto-exposure drifting mid-walk
  makes photometric optimization worse.
- 4K, 30 fps. `.MOV` or `.mp4`, any codec -- Stage 01 decodes HEVC via PyAV.
- Shoot **landscape**. A portrait clip carries a rotation flag PyAV does not
  apply; if frames extract sideways, re-run Stage 01 with `--rotate 90`.

## The walk

- Move **slowly**. Motion blur is the enemy; a slow walk in good light has
  almost none.
- Three loops around the object in one take:
  1. eye level with the kratib
  2. a second loop from **higher**, angled down
  3. a third loop from **lower**, angled up
- Keep the kratib roughly centred and fully in frame the whole time.
- ~90-120 seconds total. At the default 2 fps sampling that is ~180-240
  frames before the manual blur cull.

## Do not

- **Do not spin the kratib on a turntable.** A rotating object with a
  static background produces a nonsense reconstruction -- COLMAP cannot
  tell the object turned rather than the camera.
- Do not stop and start the recording -- one continuous take keeps the
  frames in spatial order.

## After capture

Save the file as `data/kratib.MOV` (or `.mp4`) in this repo, then:

```bat
python src\01_extract_frames.py
```
