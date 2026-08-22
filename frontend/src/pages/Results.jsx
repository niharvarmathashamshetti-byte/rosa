import React, { useEffect, useState } from 'react';
import { fetchResults } from '../services/api';
import ThreeDViewer from '../components/ThreeDViewer/ThreeDViewer';
import { ArrowLeft, Cpu, Box, Compass, AlertCircle } from 'lucide-react';

export default function Results({ caseId, onBack }) {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetchResults(caseId)
      .then((data) => {
        setResults(data);
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [caseId]);

  if (loading) {
    return <div className="card" style={{ padding: '3rem', textAlign: 'center' }}>Loading results...</div>;
  }

  if (error || !results) {
    return (
      <div className="card">
        <div className="alert alert-rose">{error || 'Results not found'}</div>
        <button className="btn btn-secondary" onClick={onBack}>
          <ArrowLeft size={16} /> Back to Case
        </button>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <button
            className="btn btn-secondary"
            onClick={onBack}
            style={{ marginBottom: '0.75rem', padding: '0.35rem 0.65rem', fontSize: '0.8rem' }}
          >
            <ArrowLeft size={14} /> Back to Case {caseId}
          </button>
          <h1 className="page-title">Surgical Planning & Reconstruction Results</h1>
          <p className="page-desc">Comprehensive analysis for patient {caseId}</p>
        </div>
      </div>

      {/* 3 Major Sections: AI Segmentation, 3D Reconstruction, Planning Parameters */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        
        {/* Section 1: AI Model Status */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <Cpu size={20} color="#f59e0b" />
            <h3 style={{ fontSize: '1.1rem', fontWeight: '600' }}>1. AI Bone Segmentation</h3>
          </div>

          <div className="alert alert-amber">
            <AlertCircle size={18} style={{ flexShrink: 0 }} />
            <div>
              <strong>Model Status: NOT TRAINED</strong>
              <div style={{ marginTop: '0.25rem' }}>{results.segmentation?.model?.message}</div>
            </div>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Once the nnU-Net multi-class knee model is trained and placed into <code>src/model/</code>, anatomical masks for Femur, Tibia, Patella, and Cartilage will automatically populate here.
          </p>
        </div>

        {/* Section 2: 3D Reconstruction */}
        <div>
          <ThreeDViewer caseId={caseId} meshes={results.reconstruction?.meshes || []} />
        </div>

        {/* Section 3: Surgical Planning Engine */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <Compass size={20} color="#06b6d4" />
            <h3 style={{ fontSize: '1.1rem', fontWeight: '600' }}>3. ROSA Surgical Planning & Resection Geometry</h3>
          </div>

          <div className="alert alert-blue" style={{ marginBottom: '1.5rem' }}>
            <AlertCircle size={18} style={{ flexShrink: 0 }} />
            <div>
              <strong>Planning Engine Status: {results.planning?.status?.toUpperCase()}</strong>
              <div style={{ marginTop: '0.25rem' }}>{results.planning?.message}</div>
            </div>
          </div>

          <div className="grid-4">
            {Object.entries(results.planning?.measurements || {}).map(([key, item]) => {
              const formattedName = key.replace(/_/g, ' ').toUpperCase();
              return (
                <div
                  key={key}
                  style={{
                    background: 'var(--bg-card)',
                    padding: '1rem',
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)'
                  }}
                >
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                    {formattedName}
                  </div>
                  <div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#9ca3af' }}>
                    NOT AVAILABLE
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '0.25rem' }}>
                    Requires Validated Landmarks
                  </div>
                </div>
              );
            })}
          </div>
        </div>

      </div>
    </div>
  );
}
