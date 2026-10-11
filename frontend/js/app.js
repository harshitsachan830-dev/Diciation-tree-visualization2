/**
 * Model Mind - Main Frontend Application Orchestrator
 * Connects API, Tree Renderer, Charts, and UI state.
 */

let currentDatasetId = 'iris';
let currentRunId = null;
let currentRunData = null;
let currentDatasetMeta = null;
let treeRenderer = null;

document.addEventListener('DOMContentLoaded', async () => {
  treeRenderer = new TreeRenderer('treeSvg', 'nodeInspector');
  window.treeRenderer = treeRenderer;

  initTabs();
  initEventListeners();

  // Check Backend Connection
  const health = await api.checkHealth();
  updateApiBadge(!!health);

  // Load default dataset
  await loadDataset('iris');
});

function updateApiBadge(connected) {
  const dot = document.getElementById('apiDot');
  const label = document.getElementById('apiStatusText');
  if (connected) {
    dot.className = 'api-dot connected';
    label.innerText = 'FastAPI Backend Online';
  } else {
    dot.className = 'api-dot';
    label.innerText = 'Standalone Browser Mode';
  }
}

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add('active');

      if (targetId === 'tabTree') {
        setTimeout(() => treeRenderer.fitTree(), 100);
      } else if (targetId === 'tab2D') {
        load2DChart();
      } else if (targetId === 'tab3D') {
        load3DChart();
      } else if (targetId === 'tabRules') {
        loadRules();
      } else if (targetId === 'tabReport') {
        loadReportSections();
      }
    });
  });
}

function initEventListeners() {
  // Demo dataset chips
  document.querySelectorAll('.chip[data-dataset]').forEach(chip => {
    chip.addEventListener('click', async () => {
      document.querySelectorAll('.chip[data-dataset]').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      await loadDataset(chip.getAttribute('data-dataset'));
    });
  });

  // CSV File upload input
  const fileInput = document.getElementById('csvFileInput');
  if (fileInput) {
    fileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        const meta = await api.uploadDataset(file);
        currentDatasetId = meta.dataset_id;
        currentDatasetMeta = meta;
        updateDatasetUI(meta);
        showToast(`Uploaded ${meta.name} successfully (${meta.rows} rows)`);
      } catch (err) {
        alert('Failed to upload CSV: ' + err.message);
      }
    });
  }

  // Missing Value Imputation Button
  const btnApplyImpute = document.getElementById('btnApplyImpute');
  if (btnApplyImpute) {
    btnApplyImpute.addEventListener('click', async () => {
      const strategy = document.getElementById('imputeStrategySelect').value;
      try {
        const res = await api.preprocessDataset({
          dataset_id: currentDatasetId,
          impute_strategy: strategy
        });
        currentDatasetMeta = res;
        updateDatasetUI(res);
        showToast(`Applied ${strategy.toUpperCase()} imputation!`);
      } catch (err) {
        alert('Imputation error: ' + err.message);
      }
    });
  }

  // Feature Scaling Button
  const btnApplyScale = document.getElementById('btnApplyScale');
  if (btnApplyScale) {
    btnApplyScale.addEventListener('click', async () => {
      const scaleMethod = document.getElementById('scaleMethodSelect').value;
      try {
        const res = await api.preprocessDataset({
          dataset_id: currentDatasetId,
          scale_method: scaleMethod
        });
        currentDatasetMeta = res;
        updateDatasetUI(res);
        showToast(`Applied ${scaleMethod.toUpperCase()} scaling!`);
      } catch (err) {
        alert('Scaling error: ' + err.message);
      }
    });
  }

  // Train Button
  const btnTrain = document.getElementById('btnTrain');
  if (btnTrain) {
    btnTrain.addEventListener('click', async () => {
      await trainModel();
    });
  }

  // Predict Row Button
  const btnPredict = document.getElementById('btnPredict');
  if (btnPredict) {
    btnPredict.addEventListener('click', async () => {
      await predictSample();
    });
  }
}

async function loadDataset(datasetName) {
  try {
    const datasets = await api.checkHealth() ? await (await fetch('/api/datasets')).json() : [];
    let meta = datasets.find(d => d.name === datasetName);
    if (!meta) {
      // Fallback preview
      meta = {
        dataset_id: datasetName,
        name: datasetName,
        columns: datasetName === 'iris' ? ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'] :
                 datasetName === 'titanic' ? ['survived', 'pclass', 'sex', 'age', 'sibsp', 'fare'] :
                 ['crim', 'rm', 'age', 'dis', 'price_k'],
        missing_summary: datasetName === 'titanic' ? { has_missing: true, columns_with_missing: { age: 7, fare: 1 } } : { has_missing: false }
      };
    }
    currentDatasetId = meta.dataset_id;
    currentDatasetMeta = meta;
    updateDatasetUI(meta);
    await trainModel();
  } catch (err) {
    console.error('Failed to load dataset:', err);
  }
}

function updateDatasetUI(meta) {
  // Update Target Select options
  const targetSelect = document.getElementById('targetColumnSelect');
  if (targetSelect && meta.columns) {
    // Smart auto-selection of Target Column
    const candidateNames = ['target', 'label', 'class', 'status', 'approved', 'loan_status', 'loan_approved', 'churn', 'outcome', 'default', 'y', 'species', 'survived', 'price_k', 'price', 'medv'];
    let selectedCol = meta.columns.find(c => candidateNames.includes(c.toLowerCase()));
    
    // If no candidate name matched, pick the LAST column (standard in CSV datasets), avoiding obvious ID columns
    if (!selectedCol && meta.columns.length > 0) {
      const nonIdCols = meta.columns.filter(c => !/(^id$|_id$|^id_|^uuid$|^index$)/i.test(c));
      selectedCol = nonIdCols.length > 0 ? nonIdCols[nonIdCols.length - 1] : meta.columns[meta.columns.length - 1];
    }
    
    if (selectedCol) targetSelect.value = selectedCol;
  }

  // Update Task (classification vs regression)
  const taskSelect = document.getElementById('taskSelect');
  if (taskSelect) {
    taskSelect.value = (meta.name === 'housing' || (targetSelect && targetSelect.value === 'price_k')) ? 'regression' : 'classification';
  }

  // Show/Hide Missing Values Alert Banner
  const alertContainer = document.getElementById('missingAlertContainer');
  if (alertContainer) {
    if (meta.missing_summary && meta.missing_summary.has_missing) {
      const cols = Object.entries(meta.missing_summary.columns_with_missing || {})
        .map(([k, v]) => `<b>${k}</b> (${v} missing)`)
        .join(', ');
      alertContainer.innerHTML = `
        <div class="alert-banner alert-amber">
          <span>⚠️ Missing values detected in: ${cols}. Use the imputer below to fill via Mean, Median, or Mode.</span>
        </div>
      `;
      alertContainer.style.display = 'block';
    } else {
      alertContainer.innerHTML = `
        <div class="alert-banner alert-emerald">
          <span>✓ Dataset clean: 0 missing values detected.</span>
        </div>
      `;
      alertContainer.style.display = 'block';
    }
  }

  // Update data table preview
  const previewContainer = document.getElementById('tablePreviewContainer');
  if (previewContainer && meta.preview) {
    let tableHtml = '<table class="data-table"><thead><tr>';
    meta.columns.forEach(c => tableHtml += `<th>${c}</th>`);
    tableHtml += '</tr></thead><tbody>';
    meta.preview.slice(0, 6).forEach(row => {
      tableHtml += '<tr>';
      meta.columns.forEach(c => {
        const val = row[c] === null || row[c] === undefined ? '<span style="color:#f43f5e">NaN</span>' : row[c];
        tableHtml += `<td>${val}</td>`;
      });
      tableHtml += '</tr>';
    });
    tableHtml += '</tbody></table>';
    previewContainer.innerHTML = tableHtml;
  }
}

async function trainModel() {
  const targetCol = document.getElementById('targetColumnSelect').value;
  const task = document.getElementById('taskSelect').value;
  const maxDepth = parseInt(document.getElementById('maxDepthInput').value) || 5;
  const criterion = document.getElementById('criterionSelect').value;
  const minSamplesSplit = parseInt(document.getElementById('minSamplesSplitInput').value) || 2;

  try {
    const res = await api.trainTree({
      dataset_id: currentDatasetId,
      target: targetCol,
      task: task,
      criterion: criterion,
      max_depth: maxDepth,
      min_samples_split: minSamplesSplit
    });

    currentRunId = res.run_id;
    currentRunData = res;

    // Update KPI Bar
    document.getElementById('kpiNodes').innerText = res.total_nodes;
    document.getElementById('kpiLeaves').innerText = res.leaves_count;
    document.getElementById('kpiDepth').innerText = res.max_depth;

    if (task === 'classification') {
      document.getElementById('kpiPurityTitle').innerText = 'TEST ACCURACY';
      document.getElementById('kpiPurity').innerText = (res.metrics.test_accuracy * 100).toFixed(1) + '%';
      document.getElementById('kpiAccuracyTitle').innerText = 'WEIGHTED F1';
      document.getElementById('kpiAccuracy').innerText = (res.metrics.f1_score * 100).toFixed(1) + '%';
    } else {
      document.getElementById('kpiPurityTitle').innerText = 'R² SCORE';
      document.getElementById('kpiPurity').innerText = res.metrics.r2_score;
      document.getElementById('kpiAccuracyTitle').innerText = 'RMSE LOSS';
      document.getElementById('kpiAccuracy').innerText = res.metrics.rmse;
    }

    // Load Tree Graph
    const treeGraph = await api.getTreeGraph(currentRunId);
    treeRenderer.renderTree(treeGraph);

    // Build Prediction Explorer Inputs
    buildPredictionInputs(treeGraph.encoded_features);

    // Render Feature Importance
    charts.renderFeatureImportance('importanceChart', res.feature_importances);

    // If active tab is 2D, 3D, or Evaluation, reload it
    const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-tab');
    if (activeTab === 'tab2D') load2DChart();
    if (activeTab === 'tab3D') load3DChart();
    if (activeTab === 'tabEval') loadEvaluationMetrics(res.metrics);

  } catch (err) {
    let msg = err.message;
    try {
      const parsed = JSON.parse(msg);
      if (parsed.detail) msg = parsed.detail;
    } catch(e) {}
    alert('Training Notice: ' + msg);
  }
}

async function load2DChart() {
  if (!currentRunId) return;
  const data = await api.get2DBoundary(currentRunId);
  charts.render2DBoundary('chart2DContainer', data);
}

async function load3DChart() {
  if (!currentRunId) return;
  const data = await api.get3DFeatureSpace(currentRunId);
  charts.render3DFeatureSpace('chart3DContainer', data);
}

function loadEvaluationMetrics(metrics) {
  if (!metrics) return;
  if (metrics.task === 'classification') {
    charts.renderConfusionMatrix('cmChartContainer', metrics.confusion_matrix, metrics.class_names);
  }
}

async function loadRules() {
  if (!currentRunId) return;
  const rules = await api.getRules(currentRunId);
  const container = document.getElementById('rulesListContainer');
  if (!container) return;

  container.innerHTML = rules.map((r, i) => `
    <div class="rule-card">
      <span style="color:var(--accent-emerald-light); font-weight:bold;">Rule #${i+1}:</span> ${r.condition}
      <div class="rule-meta">
        <span><b>Samples:</b> ${r.samples}</span>
        <span><b>Coverage:</b> ${r.coverage_pct}%</span>
        <span><b>Purity:</b> ${r.purity_pct}%</span>
      </div>
    </div>
  `).join('');
}

function buildPredictionInputs(features) {
  const container = document.getElementById('predictionInputsContainer');
  if (!container || !features) return;

  container.innerHTML = features.slice(0, 6).map(f => `
    <div class="form-group">
      <label>${f}</label>
      <input type="number" step="any" class="predict-input-field" data-feature="${f}" value="2.5" />
    </div>
  `).join('');
}

async function predictSample() {
  if (!currentRunId) return;
  const inputs = document.querySelectorAll('.predict-input-field');
  const features = {};
  inputs.forEach(inp => {
    features[inp.getAttribute('data-feature')] = parseFloat(inp.value) || 0.0;
  });

  const res = await api.predict(currentRunId, features);

  // Highlight golden animated path on the tree
  treeRenderer.highlightPath(res.path_node_ids);

  // Display prediction result
  const resultContainer = document.getElementById('predictionResultDisplay');
  if (resultContainer) {
    resultContainer.innerHTML = `
      <div style="background:var(--bg-card); padding:16px; border-radius:8px; border:1px solid var(--accent-emerald);">
        <h4 style="color:var(--accent-emerald-light); margin-bottom:8px;">🎯 Output: ${res.prediction}</h4>
        <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:12px;">Terminal Leaf Node: #${res.leaf_id}</div>
        <div style="font-size:0.8rem; font-weight:600; margin-bottom:6px;">Decision Path Breadcrumbs:</div>
        <div style="font-size:0.8rem; line-height:1.6; font-family:monospace; background:rgba(0,0,0,0.3); padding:10px; border-radius:6px;">
          ${res.path_steps.map(s => s.leaf ? `➡️ <b>Leaf Node #${s.node_id}</b> (Samples: ${s.samples})` : `↳ Node #${s.node_id}: [${s.feature} = ${s.value}] ${s.operator} ${s.threshold} (Took ${s.branch})`).join('<br>')}
        </div>
      </div>
    `;
  }
}

async function loadReportSections() {
  const sections = await api.getReport();
  const container = document.getElementById('reportViewerContainer');
  if (!container || !sections) return;

  container.innerHTML = sections.map(s => `
    <div style="background:var(--bg-secondary); border:1px solid var(--border-subtle); border-radius:8px; padding:20px; margin-bottom:20px;">
      <h3 style="color:var(--accent-emerald-light); margin-bottom:12px;">${s.title}</h3>
      <div style="font-size:0.9rem; color:var(--text-secondary); line-height:1.6;">
        ${(s.items || []).map(item => `<p style="margin-bottom:8px;">${item}</p>`).join('')}
      </div>
    </div>
  `).join('');
}

function showToast(msg) {
  console.log('Toast:', msg);
}
