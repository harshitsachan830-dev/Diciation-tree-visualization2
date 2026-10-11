/**
 * Model Mind - Decision Tree Studio API Client
 * Interfaces with FastAPI backend (Section 15 API contracts).
 */

const API_BASE = '/api';

const api = {
  isBackendConnected: false,

  async checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
      if (res.ok) {
        this.isBackendConnected = true;
        return await res.json();
      }
    } catch (e) {
      this.isBackendConnected = false;
    }
    return null;
  },

  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/datasets`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async preprocessDataset(params) {
    const res = await fetch(`${API_BASE}/datasets/preprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async trainTree(trainConfig) {
    const res = await fetch(`${API_BASE}/models/decision-tree/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(trainConfig)
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async getTreeGraph(runId) {
    const res = await fetch(`${API_BASE}/runs/${runId}/tree`);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async getNodeDetails(runId, nodeId) {
    const res = await fetch(`${API_BASE}/runs/${runId}/nodes/${nodeId}`);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async predict(runId, features) {
    const res = await fetch(`${API_BASE}/runs/${runId}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async get2DBoundary(runId, featX, featY) {
    let url = `${API_BASE}/runs/${runId}/boundary-2d`;
    const params = new URLSearchParams();
    if (featX) params.append('feat_x', featX);
    if (featY) params.append('feat_y', featY);
    if (params.toString()) url += `?${params.toString()}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async get3DFeatureSpace(runId, featX, featY, featZ) {
    let url = `${API_BASE}/runs/${runId}/feature-space-3d`;
    const params = new URLSearchParams();
    if (featX) params.append('feat_x', featX);
    if (featY) params.append('feat_y', featY);
    if (featZ) params.append('feat_z', featZ);
    if (params.toString()) url += `?${params.toString()}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async getRules(runId) {
    const res = await fetch(`${API_BASE}/runs/${runId}/rules`);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async getReport() {
    const res = await fetch(`${API_BASE}/report`);
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }
};

window.api = api;
