import React, { useState } from 'react';
import { uploadCaseFile } from '../services/api';
import { UploadCloud, CheckCircle, AlertTriangle, FileText, ArrowRight } from 'lucide-react';

export default function Upload({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setError(null);

    try {
      const res = await uploadCaseFile(file);
      setUploadResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Upload CT / Segmentation Case</h1>
          <p className="page-desc">Upload patient CT volumes or 3D segmentation data for validation and surgical planning.</p>
        </div>
      </div>

      <div className="alert alert-amber">
        <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong>Privacy Notice:</strong> All data is stored locally. Internal case IDs are automatically generated to maintain patient-level de-identification.
        </div>
      </div>

      {uploadResult ? (
        <div className="card" style={{ textAlign: 'center', padding: '3rem 2rem' }}>
          <CheckCircle size={54} color="#10b981" style={{ margin: '0 auto 1rem' }} />
          <h2 style={{ fontSize: '1.4rem', fontWeight: '700', marginBottom: '0.5rem' }}>Upload & Validation Successful!</h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
            Case <strong>{uploadResult.case_id}</strong> has been registered and validated.
          </p>

          <div style={{
            background: 'var(--bg-card)',
            padding: '1rem',
            borderRadius: '8px',
            maxWidth: '400px',
            margin: '0 auto 2rem',
            textAlign: 'left',
            fontSize: '0.85rem'
          }}>
            <div><strong>Format:</strong> {uploadResult.input_format}</div>
            <div><strong>Dimensions:</strong> {uploadResult.image_size ? uploadResult.image_size.join(' × ') : 'N/A'}</div>
            <div><strong>Status:</strong> {uploadResult.status}</div>
          </div>

          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            <button
              className="btn btn-secondary"
              onClick={() => { setFile(null); setUploadResult(null); }}
            >
              Upload Another Case
            </button>
            <button
              className="btn btn-primary"
              onClick={() => onUploadSuccess(uploadResult.case_id)}
            >
              Proceed to Case Inspection <ArrowRight size={16} />
            </button>
          </div>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="card">
          <div
            className={`dropzone ${dragActive ? 'active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            <UploadCloud size={48} color="#3b82f6" style={{ margin: '0 auto 1rem' }} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '0.5rem' }}>
              {file ? file.name : 'Click to select or drag & drop CT dataset'}
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1rem' }}>
              Supported: <strong>.zip (DICOM series)</strong>, <strong>.nii / .nii.gz</strong>, <strong>.npz</strong>, <strong>.nrrd</strong>, <strong>.mha</strong>
            </p>
            {file && (
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.35rem 0.75rem',
                background: 'rgba(59, 130, 246, 0.15)',
                borderRadius: '6px',
                color: '#60a5fa',
                fontSize: '0.85rem'
              }}>
                <FileText size={16} />
                {(file.size / (1024 * 1024)).toFixed(2)} MB
              </div>
            )}
            <input
              id="file-input"
              type="file"
              onChange={handleFileChange}
              style={{ display: 'none' }}
              accept=".zip,.nii,.gz,.npz,.nrrd,.mha"
            />
          </div>

          {error && (
            <div className="alert alert-rose" style={{ marginTop: '1.5rem' }}>
              {error}
            </div>
          )}

          <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'flex-end' }}>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={!file || uploading}
              style={{ minWidth: '160px' }}
            >
              {uploading ? 'Validating & Uploading...' : 'Upload & Validate Study'}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
