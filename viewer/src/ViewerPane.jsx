import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

function formatFor(name) {
  const n = name.toLowerCase();
  if (n.endsWith('.ply')) return GaussianSplats3D.SceneFormat.Ply;
  if (n.endsWith('.splat')) return GaussianSplats3D.SceneFormat.Splat;
  if (n.endsWith('.ksplat')) return GaussianSplats3D.SceneFormat.KSplat;
  return null;
}

const ViewerPane = forwardRef(function ViewerPane({ label, bgColor, onInteractingChange }, ref) {
  const hostRef = useRef(null);
  const viewerRef = useRef(null);
  const initialRef = useRef({ position: null, target: null });

  const [status, setStatus] = useState('empty'); // empty | loading | error | ready
  const [error, setError] = useState('');
  const [fileName, setFileName] = useState('');

  function applyBackground(hex) {
    const viewer = viewerRef.current;
    if (viewer && viewer.renderer) viewer.renderer.setClearColor(new THREE.Color(hex), 1);
  }

  function resizeToHost() {
    const viewer = viewerRef.current;
    const el = hostRef.current;
    if (!viewer || !viewer.camera || !viewer.renderer || !el) return;
    const w = el.clientWidth;
    const h = el.clientHeight;
    if (w === 0 || h === 0) return;
    viewer.camera.aspect = w / h;
    viewer.camera.updateProjectionMatrix();
    viewer.renderer.setSize(w, h, false);
  }

  async function load(file) {
    const format = formatFor(file.name);
    if (format === null) { setStatus('error'); setError('Unsupported file: ' + file.name); return; }
    setStatus('loading');
    setError('');
    setFileName(file.name);

    if (viewerRef.current) {
      try { await viewerRef.current.dispose(); } catch { /* already gone */ }
      viewerRef.current = null;
    }

    const viewer = new GaussianSplats3D.Viewer({
      cameraUp: [0, -1, 0],
      initialCameraPosition: [0, 0, -4],
      initialCameraLookAt: [0, 0, 0],
      sharedMemoryForWorkers: false,
      dynamicScene: false,
      rootElement: hostRef.current,
    });

    const url = URL.createObjectURL(file);
    try {
      await viewer.addSplatScene(url, { format, showLoadingUI: true, progressiveLoad: true });
      viewer.start();
      viewerRef.current = viewer;

      applyBackground(bgColor);
      resizeToHost();

      initialRef.current.position = viewer.camera.position.clone();
      initialRef.current.target = viewer.controls
        ? viewer.controls.target.clone()
        : new THREE.Vector3(0, 0, 0);

      const canvas = viewer.renderer.domElement;
      canvas.addEventListener('pointerdown', () => onInteractingChange(true));
      window.addEventListener('pointerup', () => onInteractingChange(false));

      setStatus('ready');
    } catch (e) {
      setStatus('error');
      setError('Failed to load:\n' + (e && e.message ? e.message : e));
      console.error(e);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  useEffect(() => { applyBackground(bgColor); }, [bgColor]);

  useEffect(() => {
    const el = hostRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => resizeToHost());
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    return () => {
      if (viewerRef.current) {
        try { viewerRef.current.dispose(); } catch { /* ignore */ }
      }
    };
  }, []);

  useImperativeHandle(ref, () => ({
    orbitStep(deltaTheta) {
      const viewer = viewerRef.current;
      if (!viewer || !viewer.camera || !viewer.controls) return;
      const { camera, controls } = viewer;
      const offset = camera.position.clone().sub(controls.target);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      spherical.theta += deltaTheta;
      offset.setFromSpherical(spherical);
      camera.position.copy(controls.target).add(offset);
      controls.update();
    },
    resetCamera() {
      const viewer = viewerRef.current;
      const initial = initialRef.current;
      if (viewer && viewer.camera && viewer.controls && initial.position) {
        viewer.camera.position.copy(initial.position);
        viewer.controls.target.copy(initial.target);
        viewer.controls.update();
      }
    },
  }));

  function onFileChange(e) {
    if (e.target.files[0]) load(e.target.files[0]);
  }
  function onDrop(e) {
    e.preventDefault();
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (f) load(f);
  }
  function onDragOver(e) { e.preventDefault(); }

  return (
    <div className="view" onDrop={onDrop} onDragOver={onDragOver}>
      <div className="viewLabel">{label}{fileName ? ` · ${fileName}` : ''}</div>
      <div className="viewerHost" ref={hostRef} />
      {status !== 'ready' && (
        <div className="drop">
          <p>Drop a <code>.ply</code>, <code>.splat</code>, or <code>.ksplat</code> file here.</p>
          <label>
            Choose file
            <input type="file" accept=".ply,.splat,.ksplat" onChange={onFileChange} />
          </label>
          {status === 'loading' && <p className="loading">Loading…</p>}
          {status === 'error' && <p className="err">{error}</p>}
        </div>
      )}
    </div>
  );
});

export default ViewerPane;
