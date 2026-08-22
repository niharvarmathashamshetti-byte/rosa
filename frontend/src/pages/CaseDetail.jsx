import React, { useEffect, useState } from 'react';
import { fetchCase, runPreprocessing, runSegmentation, runReconstruction } from '../services/api';
import StatusBadge from '../components/Layout/StatusBadge';
import CTViewer from '../components/CTViewer/CTViewer';
import { ArrowLeft, Play, RefreshCw, Layers, CheckCircle2, AlertCircle, FileSearch, Sparkles, Box } from 'lucide-react';

export default function CaseDetail({ caseId, onBack, onNavigateResults }) {
  const [caseData, setCaseData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [actionMsg, setActionMsg] = useState(null);

  const loadCase = () => {
    setLoading(true);
    fetchCase(caseId)
      .then((data) => {
        setCaseData(data);
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadCase();
  }, [caseId]);

  const handlePreprocess = async () => {
    setProcessing(true);
    setActionMsg(null);
    try {
      const res = await runPreprocessing(caseId);
      setActionMsg({ type: 'success', text: res.message });
      loadCase();
    } catch (err) {
      setActionMsg({ type: 'error', text: err.message });
    } finally {
      setProcessing(false);
    }
  };

  const handleSegment = async () => {
    setProcessing(true);
    setActionMsg(null);
    try {
      const res = await runSegmentation(caseId);
      setActionMsg({ type: 'info', text: res.message });
      loadCase();
    } catch (err) {
      setActionMsg({ type: 'error', text: err.message });
    } finally {
      setProcessing(false);
    }
  };

  const handleReconstruct = async () => {
    setProcessing(true);
    setActionMsg(null);
    try {
      const res = await runReconstruction(caseId);
      setActionMsg({ type: 'success', text: res.message });
      loadCase();
    } catch (err) {
      setActionMsg({ type: 'error', text: err.message });
    } finally {
      setProcessing(false);
    }
  };

  if (loading) {
    return <div className="card" style={{ padding: '3rem', textAlign: 'center' }}>Loading case details...</div>;
  }

  if (error || !caseData) {
    return (
      <div className="card">
        <div className="alert alert-rose">{error || 'Case not found'}</div>
        <button className="btn btn-secondary" onClick={onBack}>
          <ArrowLeft size={16} /> Back to Cases
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
            <ArrowLeft size={14} /> Back to Dashboard
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <h1 className="page-title">{caseData.case_id}</h1>
            <StatusBadge status={caseData.status} />
          </div>
          <p className="page-desc">Uploaded: {new Date(caseData.created_at).toLocaleString()}</p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            className="btn btn-primary"
            onClick={() => onNavigateResults(caseId)}
          >
            <Sparkles size={16} />
            View Results & Planning
          </button>
        </div>
      </div>

      {actionMsg && (
        <div className={`alert ${actionMsg.type === 'success' ? 'alert-blue' : actionMsg.type === 'info' ? 'alert-amber' : 'alert-rose'}`}>
          {actionMsg.text}
        </div>
      )}

      {/* Metadata Overview Card */}
      <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
        <div className="card">
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>FORMAT & VOXEL METRICS</div>
          <div style={{ fontSize: '1.1rem', fontWeight: '600', color: '#fff', marginBottom: '0.5rem' }}>
            {caseData.input_format} Volume
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            <div><strong>Voxel Spacing:</strong> {caseData.voxel_spacing ? caseData.voxel_spacing.map(s => s.toFixed(2)).join(' × ') + ' mm' : 'UNKNOWN'}</div>
            <div><strong>Matrix Shape:</strong> {caseData.image_size ? caseData.image_size.join(' × ') : 'UNKNOWN'}</div>
          </div>
        </div>

        <div className="card">
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>PHYSICAL EXTENT</div>
          <div style={{ fontSize: '1.1rem', fontWeight: '600', color: '#fff', marginBottom: '0.5rem' }}>
            {caseData.physical_size ? `${caseData.physical_size[0]} × ${caseData.physical_size[1]} × ${caseData.physical_size[2]} mm` : 'No Spatial Metadata'}
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            <div><strong>Processing:</strong> {caseData.processing_status}</div>
            <div><strong>Orientation:</strong> {caseData.orientation ? 'Standard Matrix' : 'Unspecified'}</div>
          </div>
        </div>

        <div className="card">
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>PIPELINE STAGES</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', marginTop: '0.5rem', fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <CheckCircle2 size={14} color="#10b981" /> Upload & QC Validated
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {caseData.has_preprocessed ? <CheckCircle2 size={14} color="#10b981" /> : <AlertCircle size={14} color="#f59e0b" />}
              Preprocessed Volume
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {caseData.has_meshes ? <CheckCircle2 size={14} color="#10b981" /> : <AlertCircle size={14} color="#6b7280" />}
              3D Mesh Reconstruction
            </div>
          </div>
        </div>
      </div>

      {/* Action Pipeline Toolbar */}
      <div className="card" style={{ marginBottom: '1.5rem', background: 'rgba(17, 24, 39, 0.6)' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: '600', marginBottom: '0.75rem', color: 'var(--text-muted)' }}>
          EXECUTE PIPELINE MODULES
        </h3>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
          <button
            className="btn btn-secondary"
            onClick={handlePreprocess}
            disabled={processing}
          >
            <Play size={14} />
            1. Run Standardized Preprocessing (0.5mm Isotropic)
          </button>

          <button
            className="btn btn-secondary"
            onClick={handleSegment}
            disabled={processing}
          >
            <Sparkles size={14} />
            2. Run AI Segmentation (Model Check)
          </button>

          <button
            className="btn btn-secondary"
            onClick={handleReconstruct}
            disabled={processing}
          >
            <Box size={14} />
            3. Run 3D Mesh Reconstruction (Marching Cubes)
          </button>
        </div>
      </div>

      {/* 2D Slice Viewer Component */}
      <div style={{ marginBottom: '1.5rem' }}>
        <CTViewer caseId={caseId} />
      </div>
    </div>
  );
}
