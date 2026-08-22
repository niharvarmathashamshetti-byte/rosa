import React, { useEffect, useState } from 'react';
import { fetchCases, deleteCase } from '../services/api';
import StatusBadge from '../components/Layout/StatusBadge';
import { FolderPlus, Trash2, ArrowRight, Activity, Database, AlertCircle } from 'lucide-react';

export default function Dashboard({ onSelectCase, onNavigateUpload }) {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadData = () => {
    setLoading(true);
    fetchCases()
      .then((data) => {
        setCases(data);
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleDelete = async (e, caseId) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete case '${caseId}'?`)) return;
    try {
      await deleteCase(caseId);
      loadData();
    } catch (err) {
      alert(`Failed to delete case: ${err.message}`);
    }
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Case Overview</h1>
          <p className="page-desc">Manage and inspect uploaded knee CT scans, masks, and reconstructions.</p>
        </div>
        <button className="btn btn-primary" onClick={onNavigateUpload}>
          <FolderPlus size={16} />
          Upload New Study
        </button>
      </div>

      {/* Metrics Cards */}
      <div className="grid-3" style={{ marginBottom: '2rem' }}>
        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ padding: '0.85rem', background: 'rgba(59, 130, 246, 0.15)', borderRadius: '10px' }}>
            <Database size={24} color="#60a5fa" />
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Total Cases</div>
            <div style={{ fontSize: '1.75rem', fontWeight: '700', color: '#fff' }}>{cases.length}</div>
          </div>
        </div>

        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ padding: '0.85rem', background: 'rgba(16, 185, 129, 0.15)', borderRadius: '10px' }}>
            <Activity size={24} color="#34d399" />
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Preprocessed / Valid</div>
            <div style={{ fontSize: '1.75rem', fontWeight: '700', color: '#fff' }}>
              {cases.filter(c => ['validated', 'preprocessed', 'reconstructed'].includes(c.status)).length}
            </div>
          </div>
        </div>

        <div className="card" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ padding: '0.85rem', background: 'rgba(245, 158, 11, 0.15)', borderRadius: '10px' }}>
            <AlertCircle size={24} color="#fbbf24" />
          </div>
          <div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>AI Model Status</div>
            <div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#fbbf24' }}>NOT TRAINED</div>
          </div>
        </div>
      </div>

      {/* Cases Table */}
      <div className="card">
        <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '1rem' }}>Patient CT Studies</h3>

        {loading ? (
          <p style={{ color: 'var(--text-muted)', padding: '2rem 0', textAlign: 'center' }}>Loading studies...</p>
        ) : error ? (
          <div className="alert alert-rose">{error}</div>
        ) : cases.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '3rem 1rem' }}>
            <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>No patient cases uploaded yet.</p>
            <button className="btn btn-secondary" onClick={onNavigateUpload}>
              Upload your first DICOM or NIfTI / NPZ scan
            </button>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Case ID</th>
                <th>Format</th>
                <th>Dimensions</th>
                <th>Status</th>
                <th>Created At</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {cases.map((c) => (
                <tr
                  key={c.case_id}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onSelectCase(c.case_id)}
                >
                  <td style={{ fontWeight: '600', color: '#fff' }}>{c.case_id}</td>
                  <td><span className="badge badge-gray">{c.input_format}</span></td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
                    {c.image_size ? c.image_size.join(' × ') : '—'}
                  </td>
                  <td><StatusBadge status={c.status} /></td>
                  <td style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    {new Date(c.created_at).toLocaleString()}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <div style={{ display: 'inline-flex', gap: '0.5rem' }}>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: '0.35rem 0.65rem', fontSize: '0.8rem' }}
                        onClick={(e) => { e.stopPropagation(); onSelectCase(c.case_id); }}
                      >
                        Inspect <ArrowRight size={14} />
                      </button>
                      <button
                        className="btn btn-danger"
                        style={{ padding: '0.35rem 0.65rem', fontSize: '0.8rem' }}
                        onClick={(e) => handleDelete(e, c.case_id)}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
