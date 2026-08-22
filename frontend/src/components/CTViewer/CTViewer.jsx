import React, { useState, useEffect } from 'react';
import { fetchVolumeInfo, getSliceImageUrl } from '../../services/api';
import { Sliders, Eye, AlertCircle } from 'lucide-react';

export default function CTViewer({ caseId }) {
  const [volInfo, setVolInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Viewer state
  const [axis, setAxis] = useState(0); // 0=Axial, 1=Coronal, 2=Sagittal
  const [sliceIndex, setSliceIndex] = useState(0);
  const [windowCenter, setWindowCenter] = useState(400); // Bone window level
  const [windowWidth, setWindowWidth] = useState(1500);  // Bone window width

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchVolumeInfo(caseId)
      .then((info) => {
        if (!isMounted) return;
        setVolInfo(info);
        if (info.available) {
          const midZ = Math.floor(info.axial_slices / 2);
          setSliceIndex(midZ);
        }
      })
      .catch((err) => {
        if (isMounted) setError(err.message);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [caseId]);

  if (loading) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <p style={{ color: 'var(--text-muted)' }}>Loading CT volume information...</p>
      </div>
    );
  }

  if (error || !volInfo?.available) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <AlertCircle size={40} color="#f59e0b" style={{ margin: '0 auto 1rem' }} />
        <h3 style={{ marginBottom: '0.5rem' }}>CT Viewer Data Not Available</h3>
        <p style={{ color: 'var(--text-muted)', maxWidth: '500px', margin: '0 auto' }}>
          {volInfo?.message || 'No viewable CT volume found for this case. Upload a DICOM series, NIfTI (.nii.gz), or NPZ file to inspect slices.'}
        </p>
      </div>
    );
  }

  const maxSlice = axis === 0 
    ? (volInfo.axial_slices - 1)
    : axis === 1 
      ? (volInfo.coronal_slices - 1)
      : (volInfo.sagittal_slices - 1);

  const safeSlice = Math.min(sliceIndex, maxSlice);

  const planeNames = ['Axial (Z)', 'Coronal (Y)', 'Sagittal (X)'];

  const imageUrl = getSliceImageUrl(caseId, axis, safeSlice, windowCenter, windowWidth);

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Eye size={18} color="#3b82f6" />
          <h3 style={{ fontSize: '1.1rem', fontWeight: '600' }}>2D Orthogonal CT Slice Viewer</h3>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {planeNames.map((name, idx) => (
            <button
              key={name}
              className={`btn btn-secondary ${axis === idx ? 'btn-primary' : ''}`}
              style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem' }}
              onClick={() => {
                setAxis(idx);
                const newMax = idx === 0 ? volInfo.axial_slices : idx === 1 ? volInfo.coronal_slices : volInfo.sagittal_slices;
                setSliceIndex(Math.floor(newMax / 2));
              }}
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 450px) 1fr', gap: '1.5rem', alignItems: 'start' }}>
        {/* Slice Image Canvas */}
        <div style={{
          background: '#000',
          borderRadius: '8px',
          overflow: 'hidden',
          aspectRatio: '1/1',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          border: '1px solid var(--border-color)',
          position: 'relative'
        }}>
          <img
            src={imageUrl}
            alt={`Slice ${safeSlice}`}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
          <div style={{
            position: 'absolute',
            bottom: '8px',
            left: '8px',
            background: 'rgba(0,0,0,0.7)',
            padding: '2px 8px',
            borderRadius: '4px',
            fontSize: '0.75rem',
            color: '#9ca3af',
            fontFamily: 'var(--font-mono)'
          }}>
            {planeNames[axis]} Slice: {safeSlice} / {maxSlice}
          </div>
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: '500', color: 'var(--text-muted)' }}>
                Slice Navigation ({safeSlice} / {maxSlice})
              </label>
            </div>
            <input
              type="range"
              min={0}
              max={maxSlice}
              value={safeSlice}
              onChange={(e) => setSliceIndex(parseInt(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-blue)', cursor: 'pointer' }}
            />
          </div>

          <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.75rem' }}>
              <Sliders size={16} color="#06b6d4" />
              <span style={{ fontSize: '0.85rem', fontWeight: '600' }}>CT Windowing (HU)</span>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
              <button
                className="btn btn-secondary"
                style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                onClick={() => { setWindowCenter(400); setWindowWidth(1500); }}
              >
                Bone Window (400/1500)
              </button>
              <button
                className="btn btn-secondary"
                style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                onClick={() => { setWindowCenter(40); setWindowWidth(400); }}
              >
                Soft Tissue (40/400)
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div>
                <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                  Center (Level): {windowCenter} HU
                </label>
                <input
                  type="range"
                  min={-1000}
                  max={1500}
                  step={10}
                  value={windowCenter}
                  onChange={(e) => setWindowCenter(parseInt(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>

              <div>
                <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                  Width: {windowWidth} HU
                </label>
                <input
                  type="range"
                  min={100}
                  max={3000}
                  step={50}
                  value={windowWidth}
                  onChange={(e) => setWindowWidth(parseInt(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>
            </div>
          </div>

          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
            Note: Interactive 2D slices are rendered by backend SimpleITK / Matplotlib pipelines to ensure zero GPU memory pressure on the client.
          </div>
        </div>
      </div>
    </div>
  );
}
