import React, { useState, useEffect } from 'react';
import Navbar from './components/Layout/Navbar';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import CaseDetail from './pages/CaseDetail';
import Results from './pages/Results';
import { fetchModelStatus } from './services/api';

export default function App() {
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [selectedCaseId, setSelectedCaseId] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);

  useEffect(() => {
    fetchModelStatus()
      .then(setModelStatus)
      .catch((err) => console.error("Error fetching model status:", err));
  }, []);

  const handleSelectCase = (caseId) => {
    setSelectedCaseId(caseId);
    setCurrentTab('case-detail');
  };

  const handleNavigateResults = (caseId) => {
    setSelectedCaseId(caseId);
    setCurrentTab('results');
  };

  return (
    <div className="app-container">
      <Navbar
        currentTab={currentTab}
        onSelectTab={(tab) => {
          setCurrentTab(tab);
          if (tab === 'dashboard' || tab === 'upload') {
            setSelectedCaseId(null);
          }
        }}
        modelStatus={modelStatus}
      />

      <main className="main-content">
        {currentTab === 'dashboard' && (
          <Dashboard
            onSelectCase={handleSelectCase}
            onNavigateUpload={() => setCurrentTab('upload')}
          />
        )}

        {currentTab === 'upload' && (
          <Upload
            onUploadSuccess={(newCaseId) => {
              setSelectedCaseId(newCaseId);
              setCurrentTab('case-detail');
            }}
          />
        )}

        {currentTab === 'case-detail' && selectedCaseId && (
          <CaseDetail
            caseId={selectedCaseId}
            onBack={() => setCurrentTab('dashboard')}
            onNavigateResults={handleNavigateResults}
          />
        )}

        {currentTab === 'results' && selectedCaseId && (
          <Results
            caseId={selectedCaseId}
            onBack={() => setCurrentTab('case-detail')}
          />
        )}
      </main>
    </div>
  );
}
