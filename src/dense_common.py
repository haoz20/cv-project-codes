"""
Garuda 3D Reconstruction — shared helpers for the dense pipeline
(Stages 7-13: scene prep -> undistort -> dense stereo -> fuse+mesh ->
texture -> export -> turntable render).

Mirrors common.py / colmap_model.py: this file has no digit prefix so the
numbered stage scripts (which can't import each other, see the note at the
top of 03_verify_matches.py) can all import it.

These stages wrap external CLI binaries (COLMAP, optionally OpenMVS) rather
than doing the CV math themselves, so what lives here is process plumbing:
finding the binaries, running them with consistent logging, detecting a CUDA
out-of-memory failure so 09_dense_stereo.py can retry at a lower resolution,
and a dependency-free PLY header reader for the point/vertex counts that
feed the paper's Results section.

See docs/dense_pipeline.md for the full pipeline reference and
docs/troubleshooting.md for what to do when a stage fails.
"""

import json
import os
import platform
import shutil
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
DEFAULT_SCENE = os.path.join(REPO_ROOT, "scene")

# Substrings COLMAP/CUDA print on an out-of-memory failure. Checked
# case-insensitively against stderr so 09_dense_stereo.py knows to retry at
# a lower --PatchMatchStereo.max_image_size instead of just dying.
OOM_MARKERS = (
    "out of memory",
    "cuda_error_out_of_memory",
    "cudaerrormemoryallocation",
    "cuda error: out of memory",
    "std::bad_alloc",
)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def find_binary(name, env_var, extra_names=None, doc_hint="docs/setup_rog.md"):
    """
    Locate an external binary (colmap, InterfaceCOLMAP, ...).

    Resolution order: explicit --*-bin CLI arg (handled by the caller before
    this is invoked as a fallback) -> environment variable -> PATH.
    Raises a clear, actionable error rather than letting subprocess produce
    a bare "file not found" traceback.
    """
    candidates = [name] + list(extra_names or [])
    if platform.system() == "Windows":
        candidates += [c + ".exe" for c in candidates]

    env_path = os.environ.get(env_var)
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        raise FileNotFoundError(
            f"{env_var}={env_path!r} is set but that file doesn't exist. "
            f"Unset it or point it at the real binary."
        )

    for c in candidates:
        found = shutil.which(c)
        if found:
            return found

    raise FileNotFoundError(
        f"Could not find '{name}' on PATH and {env_var} is not set.\n"
        f"See {doc_hint} for how to install it, then either add it to PATH, "
        f"set {env_var}=<path to executable>, or pass --colmap-bin/--openmvs-bin-dir explicitly."
    )


def is_oom_error(text):
    """True if stderr/stdout text looks like a CUDA out-of-memory failure."""
    if not text:
        return False
    low = text.lower()
    return any(marker in low for marker in OOM_MARKERS)


def run_subprocess(cmd, dry_run=False, cwd=None):
    """
    Run a command, always returning (returncode, stdout, stderr, elapsed)
    rather than raising -- callers decide what a failure means (09's retry
    loop treats an OOM failure very differently from a "binary not found").
    """
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}")
    if dry_run:
        return 0, "", "", 0.0

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def run_stage(stage_name, cmd, dry_run=False, cwd=None, extra_meta=None):
    """
    Run a single-attempt stage (everything except 09_dense_stereo.py, which
    has its own retry loop). Logs to output/<stage_name>.json and raises
    RuntimeError with the stderr tail on failure.
    """
    ensure_output_dir()
    returncode, stdout, stderr, elapsed = run_subprocess(cmd, dry_run=dry_run, cwd=cwd)

    log = {
        "stage": stage_name,
        "command": [str(c) for c in cmd],
        "dry_run": dry_run,
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 2),
        "stderr_tail": stderr[-2000:] if stderr else "",
    }
    if extra_meta:
        log.update(extra_meta)

    log_path = os.path.join(OUTPUT_DIR, f"{stage_name}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[saved] {log_path}")

    if not dry_run and returncode != 0:
        raise RuntimeError(
            f"{stage_name} failed (exit {returncode}). See {log_path} for the full log.\n"
            f"stderr tail:\n{stderr[-1000:]}\n"
            f"See docs/troubleshooting.md for known failure modes."
        )
    return log


def read_ply_header(path):
    """
    Parse a PLY file's header without any external dependency (no plyfile,
    no trimesh, no open3d needed just to count points). Works for both
    ascii and binary_little_endian bodies since the header itself is always
    plain text up to 'end_header'.

    Returns dict: format, vertex_count, face_count, has_color, properties.
    """
    info = {"format": None, "vertex_count": 0, "face_count": 0,
            "has_color": False, "properties": []}
    current_element = None

    with open(path, "rb") as f:
        line = f.readline().decode("ascii", errors="replace").strip()
        if line != "ply":
            raise ValueError(f"{path} does not look like a PLY file (got {line!r})")

        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"{path}: end_header not found (truncated file?)")
            line = raw.decode("ascii", errors="replace").strip()

            if line.startswith("format"):
                info["format"] = line.split()[1]
            elif line.startswith("element"):
                _, elem_name, count = line.split()
                current_element = elem_name
                if elem_name == "vertex":
                    info["vertex_count"] = int(count)
                elif elem_name == "face":
                    info["face_count"] = int(count)
            elif line.startswith("property") and current_element == "vertex":
                prop_name = line.split()[-1]
                info["properties"].append(prop_name)
                if prop_name in ("red", "green", "blue", "diffuse_red"):
                    info["has_color"] = True
            elif line == "end_header":
                break

    return info


def write_json(name, data):
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] {path}")
    return path
