import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { Box, Download, RefreshCw, Layers } from 'lucide-react';
import { getMeshDownloadUrl } from '../../services/api';

export default function ThreeDViewer({ caseId, meshes = [] }) {
  const mountRef = useRef(null);
  const [selectedMesh, setSelectedMesh] = useState(meshes[0]?.filename || null);
  const [loading, setLoading] = useState(false);
  const [renderError, setRenderError] = useState(null);

  useEffect(() => {
    if (meshes.length > 0 && !selectedMesh) {
      setSelectedMesh(meshes[0].filename);
    }
  }, [meshes]);

  useEffect(() => {
    if (!selectedMesh || !mountRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight || 450;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e17);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 150);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);

    mountRef.current.innerHTML = '';
    mountRef.current.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 2.5);
    dirLight1.position.set(1, 1, 1);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x3b82f6, 1.5);
    dirLight2.position.set(-1, -1, -1);
    scene.add(dirLight2);

    // Load STL
    setLoading(true);
    setRenderError(null);

    const loader = new STLLoader();
    const stlUrl = getMeshDownloadUrl(caseId, selectedMesh);

    let meshObject = null;

    loader.load(
      stlUrl,
      (geometry) => {
        geometry.computeVertexNormals();
        geometry.center();

        const material = new THREE.MeshPhongMaterial({
          color: 0x60a5fa,
          specular: 0x111111,
          shininess: 100,
          flatShading: false,
          side: THREE.DoubleSide,
        });

        meshObject = new THREE.Mesh(geometry, material);
        scene.add(meshObject);
        setLoading(false);
      },
      undefined,
      (err) => {
        console.error("Failed to load STL in Three.js:", err);
        setRenderError("Failed to parse/render STL file.");
        setLoading(false);
      }
    );

    // Simple interaction & animation loop
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const domElement = renderer.domElement;

    const onMouseDown = (e) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e) => {
      if (!isDragging || !meshObject) return;
      const deltaX = e.clientX - previousMousePosition.x;
      const deltaY = e.clientY - previousMousePosition.y;

      meshObject.rotation.y += deltaX * 0.01;
      meshObject.rotation.x += deltaY * 0.01;

      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => { isDragging = false; };

    const onWheel = (e) => {
      e.preventDefault();
      camera.position.z += e.deltaY * 0.1;
      camera.position.z = Math.max(20, Math.min(camera.position.z, 400));
    };

    domElement.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    domElement.addEventListener('wheel', onWheel, { passive: false });

    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      if (meshObject && !isDragging) {
        meshObject.rotation.y += 0.002; // Slow continuous spin
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationId);
      domElement.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      domElement.removeEventListener('wheel', onWheel);
      if (mountRef.current) mountRef.current.innerHTML = '';
      renderer.dispose();
    };
  }, [caseId, selectedMesh]);

  if (!meshes || meshes.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3.5rem 2rem' }}>
        <Box size={44} color="#6b7280" style={{ margin: '0 auto 1rem' }} />
        <h3 style={{ marginBottom: '0.5rem' }}>3D Model Not Available Yet</h3>
        <p style={{ color: 'var(--text-muted)', maxWidth: '480px', margin: '0 auto' }}>
          No reconstructed STL meshes were found. Run 3D reconstruction on a validated segmentation mask to generate and inspect interactive surface anatomy.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Box size={18} color="#06b6d4" />
          <h3 style={{ fontSize: '1.1rem', fontWeight: '600' }}>3D Interactive Anatomical Mesh Viewer</h3>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <Layers size={15} color="var(--text-muted)" />
          <select
            value={selectedMesh || ''}
            onChange={(e) => setSelectedMesh(e.target.value)}
            style={{
              background: 'var(--bg-card)',
              color: '#fff',
              border: '1px solid var(--border-color)',
              padding: '0.35rem 0.75rem',
              borderRadius: '6px',
              fontSize: '0.85rem'
            }}
          >
            {meshes.map((m) => (
              <option key={m.filename} value={m.filename}>
                {m.name} ({m.size_kb} KB)
              </option>
            ))}
          </select>

          {selectedMesh && (
            <a
              href={getMeshDownloadUrl(caseId, selectedMesh)}
              download
              className="btn btn-secondary"
              style={{ padding: '0.35rem 0.65rem', fontSize: '0.8rem' }}
            >
              <Download size={14} />
              STL
            </a>
          )}
        </div>
      </div>

      <div style={{ position: 'relative', width: '100%', height: '450px', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
        <div ref={mountRef} style={{ width: '100%', height: '100%', cursor: 'grab' }} />

        {loading && (
          <div style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(10, 14, 23, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            color: '#fff',
            fontSize: '0.9rem'
          }}>
            <RefreshCw size={18} className="spin" />
            Loading 3D Mesh Geometry...
          </div>
        )}

        {renderError && (
          <div style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(10, 14, 23, 0.85)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fb7185',
            fontSize: '0.9rem'
          }}>
            {renderError}
          </div>
        )}

        <div style={{
          position: 'absolute',
          bottom: '10px',
          right: '10px',
          background: 'rgba(0,0,0,0.6)',
          padding: '4px 10px',
          borderRadius: '4px',
          fontSize: '0.75rem',
          color: '#9ca3af',
          pointerEvents: 'none'
        }}>
          Left Click + Drag to Rotate • Scroll to Zoom
        </div>
      </div>
    </div>
  );
}
