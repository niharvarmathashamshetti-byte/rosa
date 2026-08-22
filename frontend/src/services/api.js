/**
 * API client service for ROSA Knee AI Frontend
 */

const API_BASE = '/api/v1';

export async function fetchHealth() {
  const res = await fetch('/health');
  if (!res.ok) throw new Error('Backend health check failed');
  return res.json();
}

export async function fetchModelStatus() {
  const res = await fetch(`${API_BASE}/model/status`);
  if (!res.ok) throw new Error('Failed to fetch model status');
  return res.json();
}

export async function fetchCases() {
  const res = await fetch(`${API_BASE}/cases`);
  if (!res.ok) throw new Error('Failed to fetch cases');
  return res.json();
}

export async function fetchCase(caseId) {
  const res = await fetch(`${API_BASE}/cases/${caseId}`);
  if (!res.ok) throw new Error(`Failed to fetch case ${caseId}`);
  return res.json();
}

export async function uploadCaseFile(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/cases`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ message: 'Upload failed' }));
    throw new Error(errorData.detail || errorData.message || 'Upload failed');
  }

  return res.json();
}

export async function deleteCase(caseId) {
  const res = await fetch(`${API_BASE}/cases/${caseId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete case ${caseId}`);
  return res.json();
}

export async function runPreprocessing(caseId) {
  const res = await fetch(`${API_BASE}/cases/${caseId}/preprocess`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: 'Preprocessing failed' }));
    throw new Error(err.detail || err.message || 'Preprocessing failed');
  }
  return res.json();
}

export async function runSegmentation(caseId) {
  const res = await fetch(`${API_BASE}/cases/${caseId}/segment`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: 'Segmentation failed' }));
    throw new Error(err.detail || err.message || 'Segmentation failed');
  }
  return res.json();
}

export async function runReconstruction(caseId) {
  const res = await fetch(`${API_BASE}/cases/${caseId}/reconstruct`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: 'Reconstruction failed' }));
    throw new Error(err.detail || err.message || 'Reconstruction failed');
  }
  return res.json();
}

export async function fetchResults(caseId) {
  const res = await fetch(`${API_BASE}/cases/${caseId}/results`);
  if (!res.ok) throw new Error(`Failed to fetch results for case ${caseId}`);
  return res.json();
}

export async function fetchVolumeInfo(caseId) {
  const res = await fetch(`${API_BASE}/cases/${caseId}/volume-info`);
  if (!res.ok) throw new Error(`Failed to fetch volume info for case ${caseId}`);
  return res.json();
}

export function getSliceImageUrl(caseId, axis, index, windowCenter = 400, windowWidth = 1500) {
  return `${API_BASE}/cases/${caseId}/slices/${axis}/${index}?window_center=${windowCenter}&window_width=${windowWidth}`;
}

export function getMeshDownloadUrl(caseId, meshFilename) {
  return `${API_BASE}/cases/${caseId}/mesh/${meshFilename}`;
}
