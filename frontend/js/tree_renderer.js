/**
 * Model Mind - Decision Tree SVG Renderer
 * Implements SVG layout, zoom, pan, fitTree, 12-field node inspector, and path highlight.
 */

class TreeRenderer {
  constructor(svgId, inspectorId) {
    this.svg = document.getElementById(svgId);
    this.inspector = document.getElementById(inspectorId);
    this.zoomGroup = null;
    this.currentTree = null;
    this.selectedNodeId = null;
    this.scale = 1;
    this.panX = 0;
    this.panY = 0;
    this.isPanning = false;
    this.startX = 0;
    this.startY = 0;

    this.initZoomPan();
  }

  initZoomPan() {
    this.svg.innerHTML = `
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
        </marker>
        <marker id="arrow-gold" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#fbbf24" />
        </marker>
        <filter id="glow-gold" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="glow" />
          <feComposite in="SourceGraphic" in2="glow" operator="over" />
        </filter>
      </defs>
      <g id="zoomGroup"></g>
    `;
    this.zoomGroup = this.svg.querySelector('#zoomGroup');

    // Drag to pan
    this.svg.addEventListener('mousedown', (e) => {
      if (e.target.closest('.tree-node-group')) return;
      this.isPanning = true;
      this.startX = e.clientX - this.panX;
      this.startY = e.clientY - this.panY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isPanning) return;
      this.panX = e.clientX - this.startX;
      this.panY = e.clientY - this.startY;
      this.updateTransform();
    });

    window.addEventListener('mouseup', () => {
      this.isPanning = false;
    });

    // Wheel to zoom
    this.svg.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      this.zoom(zoomFactor, e.clientX, e.clientY);
    });
  }

  updateTransform() {
    this.zoomGroup.setAttribute('transform', `translate(${this.panX}, ${this.panY}) scale(${this.scale})`);
  }

  zoom(factor, clientX, clientY) {
    const rect = this.svg.getBoundingClientRect();
    const cx = clientX !== undefined ? clientX - rect.left : rect.width / 2;
    const cy = clientY !== undefined ? clientY - rect.top : rect.height / 2;

    const newScale = Math.max(0.15, Math.min(3.5, this.scale * factor));
    this.panX = cx - (cx - this.panX) * (newScale / this.scale);
    this.panY = cy - (cy - this.panY) * (newScale / this.scale);
    this.scale = newScale;
    this.updateTransform();
  }

  renderTree(treeData) {
    this.currentTree = treeData;
    this.zoomGroup.innerHTML = '';

    if (!treeData || !treeData.nodes || treeData.nodes.length === 0) {
      this.zoomGroup.innerHTML = '<text x="100" y="100" fill="#64748b">No tree model loaded</text>';
      return;
    }

    // Layout hierarchy using D3-like tidy tree positioning
    const nodeMap = {};
    treeData.nodes.forEach(n => {
      nodeMap[n.node_id] = { ...n, children: [] };
    });

    let root = null;
    treeData.nodes.forEach(n => {
      const parentNode = treeData.nodes.find(p => p.left_child === n.node_id || p.right_child === n.node_id);
      if (parentNode) {
        nodeMap[parentNode.node_id].children.push(nodeMap[n.node_id]);
      } else if (n.node_id === 0) {
        root = nodeMap[n.node_id];
      }
    });

    if (!root) root = nodeMap[0] || Object.values(nodeMap)[0];

    // Compute coordinates
    const NODE_WIDTH = 190;
    const NODE_HEIGHT = 80;
    const LEVEL_HEIGHT = 130;
    let currentX = 50;

    function layout(node, depth) {
      node.y = depth * LEVEL_HEIGHT + 60;
      if (!node.children || node.children.length === 0) {
        node.x = currentX;
        currentX += NODE_WIDTH + 35;
      } else {
        node.children.forEach(c => layout(c, depth + 1));
        const first = node.children[0];
        const last = node.children[node.children.length - 1];
        node.x = (first.x + last.x) / 2;
      }
    }

    layout(root, 0);

    // Draw edges
    let edgesHtml = '';
    const nodeLookup = {};
    function collect(n) {
      nodeLookup[n.node_id] = n;
      if (n.children) n.children.forEach(collect);
    }
    collect(root);

    (treeData.edges || []).forEach(e => {
      const p = nodeLookup[e.source];
      const c = nodeLookup[e.target];
      if (p && c) {
        const x1 = p.x + NODE_WIDTH / 2;
        const y1 = p.y + NODE_HEIGHT;
        const x2 = c.x + NODE_WIDTH / 2;
        const y2 = c.y;
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        const edgeId = `edge-${p.node_id}-${c.node_id}`;

        edgesHtml += `
          <g class="tree-edge-group" id="${edgeId}">
            <path class="tree-edge" d="M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}" marker-end="url(#arrow)" />
            <rect x="${mx - 32}" y="${my - 10}" width="64" height="18" rx="4" fill="#0f172a" stroke="#334155" />
            <text class="edge-label" x="${mx}" y="${my + 3}">${e.condition || ''}</text>
          </g>
        `;
      }
    });

    // Draw nodes
    let nodesHtml = '';
    Object.values(nodeLookup).forEach(n => {
      const isLeaf = (n.kind === 'leaf');
      const headerText = isLeaf ? `Leaf #${n.node_id}: ${n.prediction}` : `${n.feature} ≤ ${n.threshold}`;
      const subtitleText = `Impurity: ${n.impurity} • N=${n.samples}`;
      const metaText = `Coverage: ${n.coverage_pct}%`;
      const badgeColor = isLeaf ? '#10b981' : '#38bdf8';

      nodesHtml += `
        <g class="tree-node-group" id="node-${n.node_id}" transform="translate(${n.x}, ${n.y})" onclick="treeRenderer.selectNode(${n.node_id})">
          <rect class="node-box" width="${NODE_WIDTH}" height="${NODE_HEIGHT}" />
          <rect x="0" y="0" width="${NODE_WIDTH}" height="5" rx="2" fill="${badgeColor}" />
          <text class="node-title" x="${NODE_WIDTH / 2}" y="28">${headerText}</text>
          <text class="node-subtitle" x="${NODE_WIDTH / 2}" y="48">${subtitleText}</text>
          <text class="node-meta" x="${NODE_WIDTH / 2}" y="66">${metaText}</text>
        </g>
      `;
    });

    this.zoomGroup.innerHTML = edgesHtml + nodesHtml;
    this.fitTree();
    this.selectNode(root.node_id);
  }

  fitTree() {
    const nodes = this.zoomGroup.querySelectorAll('.tree-node-group');
    if (nodes.length === 0) return;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    nodes.forEach(n => {
      const transform = n.getAttribute('transform');
      const match = /translate\(([^,]+),\s*([^)]+)\)/.exec(transform);
      if (match) {
        const x = parseFloat(match[1]);
        const y = parseFloat(match[2]);
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x + 190);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y + 80);
      }
    });

    const rect = this.svg.getBoundingClientRect();
    const treeW = maxX - minX;
    const treeH = maxY - minY;

    const scaleX = (rect.width - 80) / treeW;
    const scaleY = (rect.height - 80) / treeH;
    this.scale = Math.max(0.2, Math.min(1.0, Math.min(scaleX, scaleY)));

    this.panX = (rect.width - treeW * this.scale) / 2 - minX * this.scale;
    this.panY = 40;
    this.updateTransform();
  }

  selectNode(nodeId) {
    this.selectedNodeId = nodeId;
    this.svg.querySelectorAll('.tree-node-group').forEach(el => el.classList.remove('selected'));
    const selectedEl = this.svg.querySelector(`#node-${nodeId}`);
    if (selectedEl) selectedEl.classList.add('selected');

    if (!this.currentTree || !this.currentTree.nodes) return;
    const node = this.currentTree.nodes.find(n => n.node_id === nodeId);
    if (!node) return;

    // 12-field node inspector panel
    this.inspector.innerHTML = `
      <div class="inspector-title">
        <span>Node Details</span>
        <span class="brand-badge">${node.kind.toUpperCase()}</span>
      </div>
      <div class="info-row"><span class="info-key">1. Node ID</span><span class="info-val">#${node.node_id}</span></div>
      <div class="info-row"><span class="info-key">2. Depth</span><span class="info-val">${node.depth}</span></div>
      <div class="info-row"><span class="info-key">3. Node Type</span><span class="info-val">${node.kind === 'leaf' ? 'Terminal Leaf' : 'Decision Split'}</span></div>
      <div class="info-row"><span class="info-key">4. Split Feature</span><span class="info-val">${node.feature || 'None (Leaf)'}</span></div>
      <div class="info-row"><span class="info-key">5. Split Threshold</span><span class="info-val">${node.threshold !== undefined ? node.threshold : 'N/A'}</span></div>
      <div class="info-row"><span class="info-key">6. Impurity Metric</span><span class="info-val">${node.impurity}</span></div>
      <div class="info-row"><span class="info-key">7. Sample Count (N)</span><span class="info-val">${node.samples}</span></div>
      <div class="info-row"><span class="info-key">8. Coverage</span><span class="info-val">${node.coverage_pct}%</span></div>
      <div class="info-row"><span class="info-key">9. Left Child</span><span class="info-val">${node.left_child !== null ? '#' + node.left_child : 'None'}</span></div>
      <div class="info-row"><span class="info-key">10. Right Child</span><span class="info-val">${node.right_child !== null ? '#' + node.right_child : 'None'}</span></div>
      <div class="info-row"><span class="info-key">11. Output / Value</span><span class="info-val">${node.prediction !== undefined ? node.prediction : JSON.stringify(node.value)}</span></div>
      <div class="info-row"><span class="info-key">12. Distribution</span><span class="info-val">${node.distribution ? JSON.stringify(node.distribution) : 'N/A'}</span></div>
    `;
  }

  highlightPath(pathNodeIds) {
    this.svg.querySelectorAll('.tree-node-group').forEach(el => el.classList.remove('highlighted'));
    this.svg.querySelectorAll('.tree-edge').forEach(el => el.classList.remove('highlighted'));

    if (!pathNodeIds || pathNodeIds.length === 0) return;

    for (let i = 0; i < pathNodeIds.length; i++) {
      const nid = pathNodeIds[i];
      const nodeEl = this.svg.querySelector(`#node-${nid}`);
      if (nodeEl) nodeEl.classList.add('highlighted');

      if (i > 0) {
        const prevId = pathNodeIds[i - 1];
        const edgeGroup = this.svg.querySelector(`#edge-${prevId}-${nid}`);
        if (edgeGroup) {
          const pathEl = edgeGroup.querySelector('.tree-edge');
          if (pathEl) {
            pathEl.classList.add('highlighted');
            pathEl.setAttribute('marker-end', 'url(#arrow-gold)');
          }
        }
      }
    }
  }
}

window.TreeRenderer = TreeRenderer;
