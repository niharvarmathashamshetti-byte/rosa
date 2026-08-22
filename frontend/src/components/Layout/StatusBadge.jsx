import React from 'react';

export default function StatusBadge({ status }) {
  const norm = (status || '').toLowerCase();
  
  if (['uploaded', 'validated', 'available', 'ok', 'reconstructed'].includes(norm)) {
    return <span className="badge badge-green">{status}</span>;
  }
  if (['preprocessed', 'ready'].includes(norm)) {
    return <span className="badge badge-blue">{status}</span>;
  }
  if (['not_started', 'model_unavailable', 'unavailable', 'not_available'].includes(norm)) {
    return <span className="badge badge-amber">{status.replace('_', ' ')}</span>;
  }
  if (['error', 'validation_failed', 'failed'].includes(norm)) {
    return <span className="badge badge-rose">{status.replace('_', ' ')}</span>;
  }

  return <span className="badge badge-gray">{status || 'UNKNOWN'}</span>;
}
