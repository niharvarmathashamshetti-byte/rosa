import React from 'react';
import { Activity, FolderOpen, UploadCloud, Cpu } from 'lucide-react';

export default function Navbar({ currentTab, onSelectTab, modelStatus }) {
  return (
    <nav className="navbar">
      <a href="#" className="nav-brand" onClick={(e) => { e.preventDefault(); onSelectTab('dashboard'); }}>
        <div style={{
          width: '32px',
          height: '32px',
          borderRadius: '8px',
          background: 'linear-gradient(135deg, #06b6d4, #3b82f6)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 'bold',
          color: '#fff'
        }}>
          R
        </div>
        <div>
          ROSA <span>Knee AI</span>
          <div style={{ fontSize: '0.65rem', color: '#9ca3af', fontWeight: '400' }}>TKA Surgical Planning Engine</div>
        </div>
      </a>

      <div className="nav-links">
        <button
          className={`nav-btn ${currentTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => onSelectTab('dashboard')}
        >
          <FolderOpen size={16} />
          Cases
        </button>

        <button
          className={`nav-btn ${currentTab === 'upload' ? 'active' : ''}`}
          onClick={() => onSelectTab('upload')}
        >
          <UploadCloud size={16} />
          Upload Case
        </button>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.85rem' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          padding: '0.35rem 0.75rem',
          background: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '20px',
          border: '1px solid var(--border-color)'
        }}>
          <Cpu size={14} color="#f59e0b" />
          <span style={{ color: 'var(--text-muted)' }}>Model:</span>
          <span style={{ color: '#fbbf24', fontWeight: '600' }}>NOT TRAINED</span>
        </div>
      </div>
    </nav>
  );
}
