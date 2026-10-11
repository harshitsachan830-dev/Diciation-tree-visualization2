/**
 * Model Mind - Charts Module
 * Renders Plotly 2D Decision Boundaries, 3D Feature Space, Confusion Matrix, and Residuals.
 */

const chartColors = ['#10b981', '#f59e0b', '#38bdf8', '#8b5cf6', '#f43f5e', '#ec4899'];

const charts = {
  render2DBoundary(containerId, boundaryData) {
    const el = document.getElementById(containerId);
    if (!el || !boundaryData) return;

    const traces = [];

    // 1. Contour of decision regions
    if (boundaryData.grid_z) {
      traces.push({
        z: boundaryData.grid_z,
        x: boundaryData.grid_x,
        y: boundaryData.grid_y,
        type: 'contour',
        colorscale: [
          [0, 'rgba(16, 185, 129, 0.25)'],
          [0.5, 'rgba(245, 158, 11, 0.25)'],
          [1, 'rgba(56, 189, 248, 0.25)']
        ],
        contours: { coloring: 'heatmap', showlines: true },
        line: { width: 1.5, color: 'rgba(255,255,255,0.4)' },
        showscale: false,
        hoverinfo: 'none'
      });
    }

    // 2. Scatter points of samples
    if (boundaryData.scatter_points) {
      const classGroups = {};
      boundaryData.scatter_points.forEach(pt => {
        const key = String(pt.target);
        if (!classGroups[key]) classGroups[key] = { x: [], y: [] };
        classGroups[key].x.push(pt.x);
        classGroups[key].y.push(pt.y);
      });

      Object.keys(classGroups).forEach((c, idx) => {
        traces.push({
          x: classGroups[c].x,
          y: classGroups[c].y,
          mode: 'markers',
          type: 'scatter',
          name: c,
          marker: {
            size: 8,
            color: chartColors[idx % chartColors.length],
            line: { width: 1.5, color: '#ffffff' }
          }
        });
      });
    }

    const layout = {
      title: { text: `2D Decision Boundary: ${boundaryData.feature_x} vs ${boundaryData.feature_y}`, font: { color: '#f8fafc', size: 14 } },
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#111827',
      xaxis: { title: boundaryData.feature_x, color: '#94a3b8', gridcolor: '#1f2937' },
      yaxis: { title: boundaryData.feature_y, color: '#94a3b8', gridcolor: '#1f2937' },
      margin: { l: 50, r: 30, t: 50, b: 50 },
      legend: { font: { color: '#f8fafc' }, bgcolor: 'rgba(17,24,39,0.8)' }
    };

    Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: true });
  },

  render3DFeatureSpace(containerId, spaceData) {
    const el = document.getElementById(containerId);
    if (!el || !spaceData || !spaceData.points) return;

    const classGroups = {};
    spaceData.points.forEach(pt => {
      const key = String(pt.target);
      if (!classGroups[key]) classGroups[key] = { x: [], y: [], z: [] };
      classGroups[key].x.push(pt.x);
      classGroups[key].y.push(pt.y);
      classGroups[key].z.push(pt.z);
    });

    const traces = Object.keys(classGroups).map((c, idx) => ({
      x: classGroups[c].x,
      y: classGroups[c].y,
      z: classGroups[c].z,
      mode: 'markers',
      type: 'scatter3d',
      name: c,
      marker: {
        size: 5,
        color: chartColors[idx % chartColors.length],
        opacity: 0.85
      }
    }));

    const layout = {
      title: { text: `3D Feature Space: ${spaceData.feature_x} × ${spaceData.feature_y} × ${spaceData.feature_z}`, font: { color: '#f8fafc', size: 14 } },
      paper_bgcolor: 'transparent',
      scene: {
        xaxis: { title: spaceData.feature_x, color: '#94a3b8', backgroundcolor: '#0f172a', gridcolor: '#1e293b' },
        yaxis: { title: spaceData.feature_y, color: '#94a3b8', backgroundcolor: '#0f172a', gridcolor: '#1e293b' },
        zaxis: { title: spaceData.feature_z, color: '#94a3b8', backgroundcolor: '#0f172a', gridcolor: '#1e293b' }
      },
      margin: { l: 0, r: 0, t: 40, b: 0 },
      legend: { font: { color: '#f8fafc' }, bgcolor: 'rgba(17,24,39,0.8)' }
    };

    Plotly.newPlot(el, traces, layout, { responsive: true, displayModeBar: true });
  },

  renderConfusionMatrix(containerId, cmData, classNames) {
    const el = document.getElementById(containerId);
    if (!el || !cmData) return;

    const trace = {
      z: cmData,
      x: classNames,
      y: classNames,
      type: 'heatmap',
      colorscale: [
        [0, '#0f172a'],
        [0.5, '#065f46'],
        [1, '#10b981']
      ],
      showscale: true
    };

    const layout = {
      title: { text: 'Confusion Matrix Heatmap', font: { color: '#f8fafc', size: 14 } },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      xaxis: { title: 'Predicted Class', color: '#94a3b8' },
      yaxis: { title: 'Actual Class', color: '#94a3b8' },
      margin: { l: 60, r: 30, t: 50, b: 50 }
    };

    Plotly.newPlot(el, [trace], layout, { responsive: true });
  },

  renderFeatureImportance(containerId, importances) {
    const el = document.getElementById(containerId);
    if (!el || !importances) return;

    const feats = Object.keys(importances).reverse();
    const scores = Object.values(importances).reverse();

    const trace = {
      x: scores,
      y: feats,
      type: 'bar',
      orientation: 'h',
      marker: {
        color: '#10b981',
        line: { color: '#059669', width: 1 }
      }
    };

    const layout = {
      title: { text: 'Impurity-based Feature Importance', font: { color: '#f8fafc', size: 14 } },
      paper_bgcolor: 'transparent',
      plot_bgcolor: '#111827',
      xaxis: { title: 'Importance Score', color: '#94a3b8', gridcolor: '#1f2937' },
      yaxis: { color: '#f8fafc' },
      margin: { l: 110, r: 30, t: 50, b: 50 }
    };

    Plotly.newPlot(el, [trace], layout, { responsive: true });
  }
};

window.charts = charts;
