import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Visualization3D from "./Visualization3D.jsx";
import { API_BASE_URL } from "./api.js";

const CLASS_COLORS = ["#16a765", "#2583f5", "#ef4444", "#8b5cf6", "#f59e0b"];

const SECTIONS = [
  ["overview", "Overview", "⌂"],
  ["preparation", "Data Preparation", "▤"],
  ["tree", "3D Tree Visualization", "♧"],
  ["surface", "3D Surface", "▧"],
  ["boundary", "2D Decision Boundary", "⌗"],
  ["structure", "Tree Structure", "⑂"],
  ["importance", "Feature Importance", "↗"],
  ["performance", "Performance", "▥"],
  ["learning", "Learning Curve", "⌁"],
  ["confusion", "Confusion Matrix", "▦"],
  ["prediction", "Prediction", "➜"],
  ["dataset", "Dataset", "▤"],
  ["hyperparameters", "Hyperparameters", "⚙"],
  ["guide", "Decision Tree Guide", "✦"],
];

function parseCsvRecords(text) {
  const records = [];
  let record = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"' && quoted && text[index + 1] === '"') {
      value += '"';
      index += 1;
    } else if (character === '"') quoted = !quoted;
    else if (character === "," && !quoted) {
      record.push(value.trim()); value = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      record.push(value.trim());
      if (record.some((cell) => cell !== "")) records.push(record);
      record = []; value = "";
    } else value += character;
  }
  record.push(value.trim());
  if (record.some((cell) => cell !== "")) records.push(record);
  if (!records.length) return { headers: [], rows: [] };
  const headers = records[0].map((header) => header.replace(/^\uFEFF/, ""));
  return { headers, rows: records.slice(1).map((cells) => Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""]))) };
}

function isMissingCsvCell(value) {
  return ["", "na", "n/a", "null", "nan"].includes(String(value ?? "").trim().toLowerCase());
}

function percent(value) {
  return value == null ? "n/a" : `${(Number(value) * 100).toFixed(1)}%`;
}

function MetricCard({ title, value, detail, icon, tone = "green", time = false }) {
  return (
    <article className="metric-card">
      <span className={`metric-icon ${tone}`} aria-hidden="true">{icon}</span>
      <div className="metric-copy">
        <span>{title}</span>
        <strong>{time ? `${Number(value || 0).toFixed(1)} ms` : percent(value)}</strong>
        {detail && <small>{detail}</small>}
      </div>
    </article>
  );
}

function SectionHeading({ eyebrow, title, detail, action }) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <span className="section-eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
        {detail && <p>{detail}</p>}
      </div>
      {action}
    </div>
  );
}

function TreeDiagram({ nodes, maxDepth = Infinity, onSelect, selectedNode }) {
  const viewportRef = useRef(null);
  const [viewportWidth, setViewportWidth] = useState(900);
  const [viewportHeight, setViewportHeight] = useState(700);
  const [zoom, setZoom] = useState(0);
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return undefined;
    const measure = () => {
      setViewportWidth(Math.max(320, viewport.clientWidth));
      setViewportHeight(Math.max(420, viewport.clientHeight));
    };
    const observer = new ResizeObserver(measure);
    observer.observe(viewport);
    measure();
    return () => observer.disconnect();
  }, []);
  const visibleNodes = nodes.filter((node) => node.depth <= maxDepth);
  const visibleIds = new Set(visibleNodes.map((node) => node.id));
  const byId = new Map(visibleNodes.map((node) => [node.id, node]));
  const positions = new Map();
  let leafIndex = 0;

  function layout(id) {
    const node = byId.get(id);
    if (!node) return null;
    const left = visibleIds.has(node.left) ? layout(node.left) : null;
    const right = visibleIds.has(node.right) ? layout(node.right) : null;
    const x = left && right
      ? (left.x + right.x) / 2
      : left?.x ?? right?.x ?? leafIndex++ * 148 + 68;
    positions.set(id, { x, y: 35 + node.depth * 90 });
    return { x };
  }

  if (byId.has(0)) layout(0);
  const diagramWidth = Math.max(600, leafIndex * 148);
  const visibleDepth = Math.min(maxDepth, Math.max(0, ...visibleNodes.map((node) => node.depth)));
  const height = Math.max(210, (visibleDepth + 1) * 90 + 52);
  const fitScale = Math.min(1, viewportWidth / diagramWidth, viewportHeight / height);
  const displayScale = Math.min(3, Math.max(fitScale, zoom));
  const stageWidth = Math.max(viewportWidth, diagramWidth * displayScale);
  const stageHeight = Math.max(viewportHeight, height * displayScale);
  useEffect(() => {
    const viewport = viewportRef.current;
    if (viewport) viewport.scrollLeft = Math.max(0, (viewport.scrollWidth - viewport.clientWidth) / 2);
  }, [diagramWidth, displayScale]);
  return (
    <div className="tree-diagram-container">
      <div className="tree-zoom-toolbar" aria-label="Tree diagram zoom controls">
        <span>Zoom <strong>{Math.round(displayScale * 100)}%</strong></span>
        <button type="button" aria-label="Zoom out" title="Zoom out" onClick={() => setZoom((value) => Math.max(fitScale, Math.round((value - 0.25) * 100) / 100))} disabled={displayScale <= fitScale}>−</button>
        <button type="button" aria-label="Zoom in" title="Zoom in" onClick={() => setZoom((value) => Math.min(3, Math.round((value + 0.25) * 100) / 100))} disabled={displayScale >= 3}>+</button>
        <button type="button" onClick={() => setZoom(fitScale)}>Fit</button>
      </div>
      <div className="tree-diagram-scroll" ref={viewportRef}>
        <div className="tree-diagram-stage" style={{ width: `${stageWidth}px`, height: `${stageHeight}px` }}>
      <svg className="tree-diagram" style={{ width: `${diagramWidth}px`, height: `${height}px`, top: `${Math.max(0, (stageHeight - height * displayScale) / 2)}px`, transform: `scale(${displayScale})` }} viewBox={`0 0 ${diagramWidth} ${height}`} role="img" aria-label="Decision tree structure">
        {visibleNodes.map((node) => {
          const point = positions.get(node.id);
          if (!point) return null;
          return [node.left, node.right].map((childId, index) => {
            if (!visibleIds.has(childId)) return null;
            const child = positions.get(childId);
            return (
              <g key={`${node.id}-${childId}`}>
                <line x1={point.x} y1={point.y + 24} x2={child.x} y2={child.y - 22} className={index === 0 ? "tree-edge-left" : "tree-edge-right"} />
                <text x={(point.x + child.x) / 2} y={(point.y + child.y) / 2} className="tree-edge-label">{index === 0 ? "True" : "False"}</text>
              </g>
            );
          });
        })}
        {visibleNodes.map((node) => {
          const point = positions.get(node.id);
          if (!point) return null;
          const title = node.is_leaf ? node.prediction_name : `${node.feature} ≤ ${Number(node.threshold).toPrecision(4)}`;
          return (
            <g
              className={`tree-diagram-node ${node.is_leaf ? "leaf" : ""} ${selectedNode?.id === node.id ? "selected" : ""}`}
              key={node.id}
              transform={`translate(${point.x - 61} ${point.y - 22})`}
              onClick={() => onSelect?.(node)}
              role={onSelect ? "button" : undefined}
              tabIndex={onSelect ? 0 : undefined}
              onKeyDown={(event) => {
                if (onSelect && (event.key === "Enter" || event.key === " ")) onSelect(node);
              }}
            >
              <rect width="122" height="48" rx="9" />
              <text x="61" y="19" className="tree-node-title">{title.slice(0, 22)}</text>
              <text x="61" y="35" className="tree-node-detail">
                {node.is_leaf ? `${node.samples} samples` : `Gini ${node.gini} · ${node.samples} rows`}
              </text>
            </g>
          );
        })}
      </svg>
        </div>
      </div>
    </div>
  );
}

function DraggableNodeInformation({ node, classes }) {
  const workspaceRef = useRef(null);
  const cardRef = useRef(null);
  const dragRef = useRef(null);
  const [position, setPosition] = useState({ x: null, y: 14 });

  const beginDrag = (event) => {
    if (event.button !== 0) return;
    const workspace = workspaceRef.current;
    const card = cardRef.current;
    if (!workspace || !card) return;
    const bounds = workspace.getBoundingClientRect();
    const cardBounds = card.getBoundingClientRect();
    const x = position.x ?? bounds.width - cardBounds.width - 14;
    dragRef.current = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, x, y: position.y };
    event.currentTarget.setPointerCapture(event.pointerId);
    setPosition({ x, y: position.y });
    event.preventDefault();
  };

  const moveDrag = (event) => {
    const drag = dragRef.current;
    const workspace = workspaceRef.current;
    const card = cardRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !workspace || !card) return;
    const maxX = Math.max(0, workspace.clientWidth - card.offsetWidth);
    const maxY = Math.max(0, workspace.clientHeight - 96);
    setPosition({
      x: Math.min(maxX, Math.max(0, drag.x + event.clientX - drag.startX)),
      y: Math.min(maxY, Math.max(0, drag.y + event.clientY - drag.startY)),
    });
  };

  const endDrag = (event) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  };

  return (
    <div className="tree-inspector-workspace" ref={workspaceRef}>
      <div
        className="draggable-node-information"
        ref={cardRef}
        style={{ left: position.x == null ? "auto" : `${position.x}px`, right: position.x == null ? "14px" : "auto", top: `${position.y}px` }}
      >
        <div className="inspector-drag-header">
          <h3>Node Information</h3>
          <button type="button" className="inspector-drag-handle" onPointerDown={beginDrag} onPointerMove={moveDrag} onPointerUp={endDrag} onPointerCancel={endDrag} aria-label="Drag node information panel" title="Drag to move node information">⠿ <span>Move</span></button>
          <button type="button" className="inspector-reset-button" onClick={() => setPosition({ x: null, y: 14 })} aria-label="Reset node information position" title="Reset position">↺</button>
        </div>
        <NodeInformation node={node} classes={classes} showTitle={false} />
      </div>
    </div>
  );
}

function FeatureBars({ items, compact = false }) {
  const maxImportance = Math.max(...items.map((item) => item.importance), 0.001);
  return (
    <div className={`feature-bars ${compact ? "compact" : ""}`}>
      {items.map((item, index) => (
        <div className="feature-bar-row" key={item.feature}>
          <span className="feature-bar-name" title={item.feature}>{item.feature}</span>
          <div className="feature-bar-track">
            <div
              className="feature-bar-fill"
              style={{
                width: `${Math.max(1, (item.importance / maxImportance) * 100)}%`,
                background: ["#7657f6", "#3188ee", "#16a765", "#f59e0b"][index % 4],
              }}
            />
          </div>
          <strong>{(item.importance * 100).toFixed(1)}%</strong>
        </div>
      ))}
    </div>
  );
}

function ConfusionMatrix({ result, compact = false }) {
  const { class_names: classes, confusion_matrix: matrix } = result;
  const maxValue = Math.max(1, ...matrix.flat());
  return (
    <div className={`confusion-chart ${compact ? "compact" : ""}`}>
      <span className="matrix-axis-label vertical">Actual</span>
      <table>
        <thead>
          <tr><th /><th colSpan={classes.length}>Predicted label</th></tr>
          <tr><th /><th>{classes.map((name) => <span key={name}>{name}</span>)}</th></tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={classes[rowIndex]}>
              <th>{classes[rowIndex]}</th>
              {row.map((value, columnIndex) => (
                <td
                  key={`${rowIndex}-${columnIndex}`}
                  title={`Actual ${classes[rowIndex]}, predicted ${classes[columnIndex]}: ${value}`}
                  style={{
                    background: rowIndex === columnIndex
                      ? `rgba(38, 112, 220, ${0.13 + (value / maxValue) * 0.72})`
                      : `rgba(111, 145, 196, ${0.04 + (value / maxValue) * 0.25})`,
                  }}
                >
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <span className="matrix-axis-label horizontal">Predicted class</span>
    </div>
  );
}

function LearningCurve({ points }) {
  if (!points?.length) return <div className="chart-empty">Train a tree to calculate its learning curve.</div>;
  const width = 520;
  const height = 260;
  const left = 42;
  const right = 16;
  const top = 16;
  const bottom = 42;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const maxRows = Math.max(...points.map((point) => point.training_rows), 1);
  const x = (value) => left + (value / maxRows) * plotWidth;
  const y = (value) => top + (1 - value) * plotHeight;
  const path = (key) => points.map((point, index) => `${index ? "L" : "M"} ${x(point.training_rows)} ${y(point[key])}`).join(" ");
  return (
    <div className="learning-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training and validation learning curve">
        {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((score) => (
          <g key={score}>
            <line className="chart-grid-line" x1={left} x2={width - right} y1={y(score)} y2={y(score)} />
            <text className="chart-tick" x={left - 7} y={y(score) + 4} textAnchor="end">{score.toFixed(1)}</text>
          </g>
        ))}
        <path d={path("training_score")} className="learning-line train" />
        <path d={path("validation_score")} className="learning-line validation" />
        {points.map((point) => (
          <g key={point.training_rows}>
            <circle className="learning-dot train" cx={x(point.training_rows)} cy={y(point.training_score)} r="4" />
            <circle className="learning-dot validation" cx={x(point.training_rows)} cy={y(point.validation_score)} r="4" />
          </g>
        ))}
        <text className="chart-axis-title" x={left + plotWidth / 2} y={height - 8} textAnchor="middle">Training size (rows)</text>
      </svg>
      <div className="chart-legend">
        <span><i className="train" />Training score</span>
        <span><i className="validation" />Validation score</span>
      </div>
    </div>
  );
}

function BoundaryPlot({ result, features }) {
  if (!features[0] || !features[1]) return <div className="chart-empty">Select two numeric features in the sidebar.</div>;
  const rows = result.dataset.rows_data.filter((row) =>
    Number.isFinite(Number(row[features[0]])) &&
    Number.isFinite(Number(row[features[1]])),
  );
  if (!rows.length) return <div className="chart-empty">No numeric rows are available for this projection.</div>;

  const width = 600;
  const height = 360;
  const pad = { left: 55, right: 14, top: 14, bottom: 44 };
  const xs = rows.map((row) => Number(row[features[0]]));
  const ys = rows.map((row) => Number(row[features[1]]));
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const sx = (value) => pad.left + ((Number(value) - xMin) / (xMax - xMin || 1)) * (width - pad.left - pad.right);
  const sy = (value) => height - pad.bottom - ((Number(value) - yMin) / (yMax - yMin || 1)) * (height - pad.top - pad.bottom);
  const colors = Object.fromEntries(result.class_names.map((name, index) => [name, CLASS_COLORS[index % CLASS_COLORS.length]]));
  const defaults = Object.fromEntries(result.dataset.features.map((feature) => {
    const values = result.dataset.rows_data.map((row) => Number(row[feature])).filter(Number.isFinite).sort((a, b) => a - b);
    return [feature, values[Math.floor(values.length / 2)] ?? 0];
  }));
  const nodeMap = new Map(result.tree.map((node) => [node.id, node]));
  const cells = [];
  const columns = 24;
  const rowCount = 15;
  for (let iy = 0; iy < rowCount; iy += 1) {
    for (let ix = 0; ix < columns; ix += 1) {
      const sample = {
        ...defaults,
        [features[0]]: xMin + ((ix + 0.5) / columns) * (xMax - xMin),
        [features[1]]: yMin + (1 - (iy + 0.5) / rowCount) * (yMax - yMin),
      };
      let node = nodeMap.get(0);
      while (node && !node.is_leaf) {
        const value = Number(sample[node.feature]);
        node = nodeMap.get(value <= node.threshold ? node.left : node.right);
      }
      cells.push({
        x: pad.left + (ix / columns) * (width - pad.left - pad.right),
        y: pad.top + (iy / rowCount) * (height - pad.top - pad.bottom),
        color: colors[node?.prediction_name] || "#94a3b8",
        key: `${ix}-${iy}`,
      });
    }
  }
  return (
    <div className="boundary-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Decision boundary for ${features[0]} and ${features[1]}`}>
        <rect className="plot-background" x={pad.left} y={pad.top} width={width - pad.left - pad.right} height={height - pad.top - pad.bottom} />
        {cells.map((cell) => <rect key={cell.key} x={cell.x} y={cell.y} width={(width - pad.left - pad.right) / columns + 0.3} height={(height - pad.top - pad.bottom) / rowCount + 0.3} fill={cell.color} fillOpacity="0.17" />)}
        {result.tree.filter((node) => !node.is_leaf).map((node) => {
          const color = colors[node.prediction_name] || "#71809a";
          if (node.feature === features[0]) return <line key={node.id} className="boundary-split" x1={sx(node.threshold)} x2={sx(node.threshold)} y1={pad.top} y2={height - pad.bottom} stroke={color} />;
          if (node.feature === features[1]) return <line key={node.id} className="boundary-split" x1={pad.left} x2={width - pad.right} y1={sy(node.threshold)} y2={sy(node.threshold)} stroke={color} />;
          return null;
        })}
        {rows.map((row, index) => <circle key={index} cx={sx(row[features[0]])} cy={sy(row[features[1]])} r="2.5" fill={colors[String(row[result.dataset.target])] || "#64748b"} fillOpacity="0.8" />)}
        <text className="chart-axis-title" x={(pad.left + width - pad.right) / 2} y={height - 11} textAnchor="middle">{features[0]}</text>
        <text className="chart-axis-title" transform={`translate(15 ${(pad.top + height - pad.bottom) / 2}) rotate(-90)`} textAnchor="middle">{features[1]}</text>
      </svg>
      <div className="chart-legend">
        {result.class_names.map((name) => <span key={name}><i style={{ background: colors[name] }} />{name}</span>)}
      </div>
    </div>
  );
}

function App() {
  const [section, setSection] = useState("overview");
  const [theme, setTheme] = useState(() => window.localStorage.getItem("modelmind-theme") || "light");
  const [file, setFile] = useState(null);
  const [csvPreview, setCsvPreview] = useState({ headers: [], rows: [] });
  const [imputationStrategy, setImputationStrategy] = useState("mean");
  const [target, setTarget] = useState("target");
  const [maxDepth, setMaxDepth] = useState(4);
  const [criterion, setCriterion] = useState("gini");
  const [splitter, setSplitter] = useState("best");
  const [minSamplesSplit, setMinSamplesSplit] = useState(2);
  const [minSamplesLeaf, setMinSamplesLeaf] = useState(1);
  const [maxLeafNodes, setMaxLeafNodes] = useState(0);
  const [maxFeatures, setMaxFeatures] = useState("all");
  const [classWeight, setClassWeight] = useState("none");
  const [minImpurityDecrease, setMinImpurityDecrease] = useState(0);
  const [ccpAlpha, setCcpAlpha] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState("tree");
  const [plotFeatures, setPlotFeatures] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [predictionRowIndex, setPredictionRowIndex] = useState("0");
  const [prediction, setPrediction] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);

  useEffect(() => {
    window.localStorage.setItem("modelmind-theme", theme);
  }, [theme]);

  const numericFeatures = useMemo(() => {
    if (!result) return [];
    return result.dataset.features.filter((feature) =>
      result.dataset.rows_data.some((row) => row[feature] !== "" && row[feature] != null && Number.isFinite(Number(row[feature]))),
    );
  }, [result]);

  const updatePlotFeature = (axis, feature) => {
    setPlotFeatures((current) => {
      const next = [...current];
      next[axis] = feature;
      return next;
    });
  };

  const handleFile = async (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setResult(null);
    setPrediction(null);
    setSection("preparation");
    setError("");
    try {
      const parsed = parseCsvRecords(await selected.text());
      const columns = parsed.headers;
      if (!columns.length) throw new Error("The CSV file does not have a readable header row.");
      setCsvPreview({ headers: columns, rows: parsed.rows });
      setTarget((current) => columns.includes(current) ? current : columns.at(-1));
    } catch (readError) {
      setError(`Could not read the CSV header: ${readError.message}`);
    }
  };

  const train = async (datasetFile = file, datasetTarget = target) => {
    setError("");
    if (!datasetFile) {
      setError("Choose a CSV dataset first.");
      return;
    }
    if (!datasetTarget.trim()) {
      setError("Choose the target column.");
      return;
    }
    setLoading(true);
    try {
      const body = new FormData();
      body.append("file", datasetFile);
      const query = new URLSearchParams({
        target: datasetTarget,
        max_depth: String(maxDepth),
        criterion,
        splitter,
        min_samples_split: String(minSamplesSplit),
        min_samples_leaf: String(minSamplesLeaf),
        max_leaf_nodes: String(maxLeafNodes),
        max_features: maxFeatures,
        class_weight: classWeight,
        min_impurity_decrease: String(minImpurityDecrease),
        ccp_alpha: String(ccpAlpha),
        imputation_strategy: imputationStrategy,
      });
      const response = await fetch(`${API_BASE_URL}/train?${query}`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) {
        const detail = data.detail;
        const message = Array.isArray(detail)
          ? detail.map((item) => `${item.loc?.at(-1) || "Input"}: ${item.msg}`).join(" · ")
          : typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : "Training failed.";
        throw new Error(message);
      }
      setResult(data);
      setSection("overview");
      setViewMode("tree");
      setSelectedNode(data.tree.find((node) => node.id === 0) || null);
      setPrediction(null);
      const numeric = data.dataset.features.filter((feature) =>
        data.dataset.rows_data.some((row) => row[feature] !== "" && row[feature] != null && Number.isFinite(Number(row[feature]))),
      );
      setPlotFeatures(numeric.slice(0, 3));
    } catch (trainError) {
      setError(trainError.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    async function loadDemo() {
      try {
        const response = await fetch(`${API_BASE_URL}/sample-dataset`);
        if (!response.ok) return;
        const blob = await response.blob();
        const demoFile = new File([blob], "iris.csv", { type: "text/csv" });
        if (cancelled) return;
        setFile(demoFile);
        const parsed = parseCsvRecords(await demoFile.text());
        setCsvPreview({ headers: parsed.headers, rows: parsed.rows });
        setTarget("target");
        await train(demoFile, "target");
      } catch {
        // The empty state remains useful when the API is not running yet.
      }
    }
    loadDemo();
    return () => { cancelled = true; };
  // The demo should load once when the workspace opens.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectNode = useCallback((node) => setSelectedNode(node), []);

  const tracePrediction = async () => {
    if (!result) return;
    const row = result.dataset.rows_data[Number(predictionRowIndex)];
    if (!row) {
      setError("Choose a valid dataset row.");
      return;
    }
    setPredictionLoading(true);
    setError("");
    try {
      const features = Object.fromEntries(result.dataset.features.map((feature) => [feature, row[feature]]));
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not trace this prediction.");
      setPrediction(data);
    } catch (predictionError) {
      setError(predictionError.message);
    } finally {
      setPredictionLoading(false);
    }
  };

  const downloadReport = () => {
    if (!result) return;
    const report = {
      dataset: result.dataset.filename,
      target: result.dataset.target,
      rows: result.dataset.rows,
      features: result.dataset.features,
      classes: result.class_names,
      metrics: result.metrics,
      model: result.complexity,
      feature_importance: result.feature_importance,
      confusion_matrix: result.confusion_matrix,
    };
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    link.download = `${file?.name.replace(/\.csv$/i, "") || "decision-tree"}-report.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const sectionTitle = SECTIONS.find(([id]) => id === section)?.[1] || "Overview";
  const main3DMode = section === "tree" ? "tree" : viewMode === "scatter" ? "scatter" : "surface";
  const selectedFeatures = section === "boundary" ? plotFeatures.slice(0, 2) : plotFeatures.slice(0, 3);

  return (
    <div className={`dashboard-shell ${theme === "dark" ? "theme-dark" : ""}`}>
      <aside className="sidebar">
        <div className="brand-lockup">
          <span className="brand-mark">M</span>
          <span className="brand-name">ModelMind</span>
        </div>

        <nav className="sidebar-navigation" aria-label="Model visualization sections">
          {SECTIONS.map(([id, label, icon]) => (
            <button key={id} type="button" className={`nav-item ${section === id ? "active" : ""}`} aria-current={section === id ? "page" : undefined} onClick={() => setSection(id)}>
              <span className="nav-icon" aria-hidden="true">{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <section className="sidebar-settings" aria-label="Dataset and training settings">
          <div className="sidebar-settings-title">Dataset</div>
          <input id="dataset-upload" className="visually-hidden" type="file" accept=".csv,text/csv" onChange={handleFile} />
          <label className="dataset-picker" htmlFor="dataset-upload">
            <span>{file?.name || "Choose CSV dataset"}</span>
            <span aria-hidden="true">⌄</span>
          </label>
          <label className="control-label" htmlFor="target-column">Target column</label>
          <input id="target-column" className="sidebar-input" value={target} onChange={(event) => setTarget(event.target.value)} placeholder="e.g. Result" />

          {result && (
            <>
              <div className="sidebar-settings-title feature-settings-title">Features (3D)</div>
              {[0, 1, 2].map((axis) => (
                <label className="axis-control" key={axis}>
                  <span>{["X axis", "Y axis", "Z axis"][axis]}</span>
                  <select
                    value={plotFeatures[axis] || ""}
                    onChange={(event) => updatePlotFeature(axis, event.target.value)}
                    disabled={!numericFeatures.length}
                  >
                    {numericFeatures.map((feature) => <option value={feature} key={feature}>{feature}</option>)}
                  </select>
                </label>
              ))}
            </>
          )}

          <div className="sidebar-settings-title feature-settings-title">Tree Settings</div>
          <label className="control-label" htmlFor="max-depth">Max depth</label>
          <input id="max-depth" className="sidebar-input" type="number" min="1" max="20" value={maxDepth} onChange={(event) => setMaxDepth(Math.min(20, Math.max(1, Number(event.target.value))))} />
        <label className="control-label" htmlFor="criterion">Split quality (criterion)</label>
        <select id="criterion" className="sidebar-input" value={criterion} onChange={(event) => setCriterion(event.target.value)}><option value="gini">Gini impurity</option><option value="entropy">Entropy</option><option value="log_loss">Log loss</option></select>
        <label className="control-label" htmlFor="splitter">Split strategy</label>
        <select id="splitter" className="sidebar-input" value={splitter} onChange={(event) => setSplitter(event.target.value)}><option value="best">Best split</option><option value="random">Random candidate</option></select>
        <label className="control-label" htmlFor="min-split">Minimum samples to split</label>
        <input id="min-split" className="sidebar-input" type="number" min="2" value={minSamplesSplit} onChange={(event) => setMinSamplesSplit(Math.max(2, Number(event.target.value)))} />
          <label className="control-label" htmlFor="min-leaf">Minimum samples per leaf</label>
          <input id="min-leaf" className="sidebar-input" type="number" min="1" value={minSamplesLeaf} onChange={(event) => setMinSamplesLeaf(Math.max(1, Number(event.target.value)))} />
        <label className="control-label" htmlFor="max-leaf-nodes">Maximum leaf nodes (0 = unlimited)</label>
        <input id="max-leaf-nodes" className="sidebar-input" type="number" min="0" value={maxLeafNodes} onChange={(event) => setMaxLeafNodes(Math.max(0, Number(event.target.value)))} />
        <label className="control-label" htmlFor="max-features">Features considered per split</label>
        <select id="max-features" className="sidebar-input" value={maxFeatures} onChange={(event) => setMaxFeatures(event.target.value)}><option value="all">All features</option><option value="sqrt">Square root</option><option value="log2">Log₂</option></select>
        <label className="control-label" htmlFor="class-weight">Class balancing</label>
        <select id="class-weight" className="sidebar-input" value={classWeight} onChange={(event) => setClassWeight(event.target.value)}><option value="none">No weighting</option><option value="balanced">Balanced classes</option></select>
        <label className="control-label" htmlFor="min-impurity">Minimum impurity decrease</label>
        <input id="min-impurity" className="sidebar-input" type="number" min="0" step="0.001" value={minImpurityDecrease} onChange={(event) => setMinImpurityDecrease(Math.max(0, Number(event.target.value)))} />
          <label className="control-label" htmlFor="pruning-alpha">Pruning alpha</label>
          <input id="pruning-alpha" className="sidebar-input" type="number" min="0" step="0.001" value={ccpAlpha} onChange={(event) => setCcpAlpha(Math.max(0, Number(event.target.value)))} />
          <button className="update-button" type="button" onClick={() => train()} disabled={loading}>
            {loading ? "Training…" : result ? "Update Visualization" : "Train Tree"}
          </button>
        </section>
      </aside>

      <main className="dashboard-main">
        <header className="topbar">
          <div className="page-brand">
            <span className="page-brand-icon">♧</span>
            <div>
              <h1>{section === "overview" ? "Decision Tree Visualization" : section === "surface" ? (viewMode === "scatter" ? "3D Scatter" : "3D Surface") : sectionTitle}</h1>
              <p>Explore how your Decision Tree learns, splits the feature space, and makes predictions.</p>
            </div>
          </div>
          <div className="topbar-actions">
            <button className="theme-toggle" type="button" onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`} title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
              <span aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span><span>{theme === "dark" ? "Light" : "Dark"}</span>
            </button>
            {["tree", "surface", "scatter"].map((mode) => (
              <button key={mode} type="button" className={`mode-button ${viewMode === mode ? "active" : ""}`} onClick={() => {
                setViewMode(mode);
                if (mode === "tree") setSection("tree");
                else setSection("surface");
              }}>
                {mode === "tree" ? "♧  3D Tree" : mode === "surface" ? "▧  3D Surface" : "◇  3D Scatter"}
              </button>
            ))}
            <button className="download-button" type="button" onClick={downloadReport} disabled={!result}>↓ <span>Download Report</span></button>
          </div>
        </header>

        {error && <div className="error-banner" role="alert">{error}</div>}

        {!result && !["preparation", "guide"].includes(section) ? (
          <section className="welcome-panel">
            <div className="welcome-icon">◇</div>
            <span className="section-eyebrow">INTERACTIVE MODEL EXPLORER</span>
            <h2>Your Decision Tree workspace</h2>
            <p>Choose a classification CSV from the left panel, confirm its target column, and train a tree to explore the visualizations and evaluation metrics.</p>
            <button type="button" className="primary-button" onClick={() => document.getElementById("dataset-upload")?.click()}>Choose a dataset</button>
          </section>
        ) : (
          <>
            {section === "guide" && <DecisionTreeGuide maxDepth={maxDepth} setMaxDepth={setMaxDepth} result={result} file={file} loading={loading} onTrain={train} />}
            {section === "preparation" && (
              <DataPreparation
                file={file}
                preview={csvPreview}
                target={target}
                setTarget={setTarget}
                strategy={imputationStrategy}
                setStrategy={setImputationStrategy}
                onChooseFile={() => document.getElementById("dataset-upload")?.click()}
                onTrain={() => train()}
                loading={loading}
                existingResult={result}
              />
            )}
            {result && <>
            {section === "overview" && (
              <div className="overview-page">
                <section className="metric-strip" aria-label="Model performance summary">
                  <MetricCard title="Accuracy" value={result.metrics.accuracy} icon="✦" tone="green" />
                  <MetricCard title="Precision" value={result.metrics.precision} icon="▥" tone="blue" />
                  <MetricCard title="Recall" value={result.metrics.recall} icon="↗" tone="purple" />
                  <MetricCard title="F1 Score" value={result.metrics.f1} icon="↗" tone="orange" />
                  <MetricCard title="Training Time" value={result.metrics.training_time_ms} icon="◷" tone="red" time />
                </section>

                <div className="reference-hero-grid">
                  <section className="panel surface-panel reference-canvas-panel">
                    <div className="panel-heading"><div><h2>3D Decision Tree</h2><p>Each node is a split; the path and colored regions explain the model in feature space.</p></div><div className="inline-mode-switch">{["tree", "surface", "scatter"].map((mode) => <button type="button" key={mode} className={viewMode === mode ? "active" : ""} onClick={() => setViewMode(mode)}>{mode === "tree" ? "3D Tree" : mode === "surface" ? "3D Surface" : "3D Scatter"}</button>)}</div></div>
                    <Visualization3D mode={viewMode} dark={theme === "dark"} nodes={result.tree} rows={result.dataset.rows_data} features={plotFeatures} allFeatures={result.dataset.features} target={result.dataset.target} classNames={result.class_names} selectedNode={selectedNode?.id} onSelectNode={selectNode} />
                    <div className="surface-axis-legend">{result.class_names.map((name, index) => <span key={name}><i className="axis-dot" style={{ background: CLASS_COLORS[index % CLASS_COLORS.length] }} />{name}</span>)}</div>
                  </section>
                  <aside className="reference-side-rail">
                    <NodeInformation node={selectedNode} classes={result.class_names} />
                    <TreeControls
                      result={result} file={file} loading={loading} onTrain={train}
                      maxDepth={maxDepth} setMaxDepth={setMaxDepth}
                      criterion={criterion} setCriterion={setCriterion} splitter={splitter} setSplitter={setSplitter}
                      minSamplesSplit={minSamplesSplit} setMinSamplesSplit={setMinSamplesSplit}
                      minSamplesLeaf={minSamplesLeaf} setMinSamplesLeaf={setMinSamplesLeaf}
                      maxLeafNodes={maxLeafNodes} setMaxLeafNodes={setMaxLeafNodes}
                      maxFeatures={maxFeatures} setMaxFeatures={setMaxFeatures}
                      classWeight={classWeight} setClassWeight={setClassWeight}
                      minImpurityDecrease={minImpurityDecrease} setMinImpurityDecrease={setMinImpurityDecrease}
                      ccpAlpha={ccpAlpha} setCcpAlpha={setCcpAlpha}
                      features={plotFeatures}
                    />
                  </aside>
                </div>

                <div className="overview-bottom-grid">
                  <section className="panel chart-panel"><div className="panel-heading"><div><h2>2D Decision Boundary</h2><p>{plotFeatures.slice(0, 2).join(" vs ")}</p></div><button className="subtle-link" type="button" onClick={() => setSection("boundary")}>Open</button></div><BoundaryPlot result={result} features={plotFeatures} /></section>
                  <section className="panel chart-panel"><div className="panel-heading"><div><h2>Feature Importance</h2><p>Relative contribution to tree splits.</p></div><button className="subtle-link" type="button" onClick={() => setSection("importance")}>Open</button></div><FeatureBars items={result.feature_importance.slice(0, 5)} compact /></section>
                  <section className="panel chart-panel"><div className="panel-heading"><div><h2>Confusion Matrix</h2><p>Held-out test rows.</p></div><button className="subtle-link" type="button" onClick={() => setSection("confusion")}>Open</button></div><ConfusionMatrix result={result} compact /></section>
                  <section className="panel chart-panel learning-preview"><div className="panel-heading"><div><h2>Learning Curve</h2><p>Training and validation accuracy.</p></div><button className="subtle-link" type="button" onClick={() => setSection("learning")}>Open</button></div><LearningCurve points={result.learning_curve} /></section>
                </div>
                <section className="panel overview-features"><div className="features-summary"><div><h2>Dataset Overview</h2><p>{result.dataset.filename} · {result.dataset.rows} rows · {result.dataset.columns} columns · target <strong>{result.dataset.target}</strong></p></div><button className="subtle-link" type="button" onClick={() => setSection("dataset")}>Explore dataset</button></div><div className="feature-chip-list">{result.dataset.features.map((feature) => <span key={feature}>{feature}</span>)}</div></section>
              </div>
            )}

            {section === "tree" && (
              <section className="panel focused-panel tree-visualization-panel">
                <div className="panel-heading"><div><h2>3D Decision Tree</h2><p>Drag to orbit · scroll to zoom · double-click a node to inspect it.</p></div><span className="model-pill">{result.complexity.actual_depth} levels · {result.complexity.node_count} nodes</span></div>
                <Visualization3D mode="tree" dark={theme === "dark"} nodes={result.tree} classNames={result.class_names} selectedNode={selectedNode?.id} onSelectNode={selectNode} />
                {selectedNode && <NodeInformation node={selectedNode} classes={result.class_names} />}
              </section>
            )}

            {section === "surface" && (
              <section className="panel focused-panel feature-space-panel"><div className="panel-heading"><div><h2>{viewMode === "scatter" ? "3D Scatter Plot" : "3D Decision Regions"}</h2><p>{viewMode === "scatter" ? "Each sphere is one row, colored by its observed class. Drag to rotate; scroll to zoom." : "Transparent colored blocks show the model’s predicted class; spheres show observed rows."}</p></div></div><div className="surface-control-strip">{[0, 1, 2].map((axis) => <label key={axis}>{["X axis", "Y axis", "Z axis"][axis]}<select value={plotFeatures[axis] || ""} onChange={(event) => updatePlotFeature(axis, event.target.value)}>{numericFeatures.map((feature) => <option key={feature}>{feature}</option>)}</select></label>)}</div><Visualization3D mode={viewMode === "scatter" ? "scatter" : "surface"} dark={theme === "dark"} nodes={result.tree} rows={result.dataset.rows_data} features={plotFeatures} allFeatures={result.dataset.features} target={result.dataset.target} classNames={result.class_names} /><div className="surface-class-legend"><strong>{viewMode === "scatter" ? "Observed class" : "Predicted region · observed sample"}</strong>{result.class_names.map((name, index) => <span key={name}><i style={{ background: CLASS_COLORS[index % CLASS_COLORS.length] }} />{name}</span>)}</div></section>
            )}

            {section === "boundary" && (
              <section className="panel focused-panel"><SectionHeading eyebrow="FEATURE SPACE" title="2D Decision Boundary" detail="Colored regions show the model prediction; points show actual dataset rows." /><div className="surface-control-strip two-axis">{[0, 1].map((axis) => <label key={axis}>{axis === 0 ? "X feature" : "Y feature"}<select value={plotFeatures[axis] || ""} onChange={(event) => updatePlotFeature(axis, event.target.value)}>{numericFeatures.map((feature) => <option key={feature}>{feature}</option>)}</select></label>)}</div><BoundaryPlot result={result} features={plotFeatures} /></section>
            )}

            {section === "structure" && (
              <section className="panel focused-panel structure-view-panel"><SectionHeading eyebrow="MODEL RULES" title="Tree Structure" detail="Select any node to inspect its rule and class distribution. Drag the node information card out of the way to reveal more of the tree." /><div className="structure-view-grid"><TreeDiagram nodes={result.tree} onSelect={setSelectedNode} selectedNode={selectedNode} /><DraggableNodeInformation node={selectedNode || result.tree[0]} classes={result.class_names} /></div></section>
            )}

            {section === "importance" && (
              <section className="panel focused-panel"><SectionHeading eyebrow="MODEL EXPLANATION" title="Feature Importance" detail="Impurity-based importance measures how much each feature contributes to reducing split impurity." /><FeatureBars items={result.feature_importance} /></section>
            )}

            {section === "performance" && <PerformanceReport result={result} />}
            {section === "learning" && <section className="panel focused-panel"><SectionHeading eyebrow="MODEL DIAGNOSTICS" title="Learning Curve" detail="Training and held-out validation accuracy as the training set grows." /><LearningCurve points={result.learning_curve} /></section>}
            {section === "confusion" && <section className="panel focused-panel"><SectionHeading eyebrow="CLASSIFICATION" title="Confusion Matrix" detail="Rows are actual classes; columns are model predictions on the held-out test split." /><ConfusionMatrix result={result} /></section>}

            {section === "prediction" && (
              <section className="panel focused-panel"><SectionHeading eyebrow="SINGLE-ROW EXPLANATION" title="Prediction Path" detail="Choose a dataset row to replay each decision rule from the root to its prediction." /><div className="prediction-controls"><label>Dataset row<select value={predictionRowIndex} onChange={(event) => setPredictionRowIndex(event.target.value)}>{result.dataset.rows_data.map((row, index) => <option value={index} key={index}>Row {index + 1} · {String(row[result.dataset.target])}</option>)}</select></label><button className="primary-button" type="button" onClick={tracePrediction} disabled={predictionLoading}>{predictionLoading ? "Tracing…" : "Trace prediction"}</button></div>{prediction && <div className="prediction-path"><div className="prediction-result"><span>Predicted class</span><strong>{prediction.prediction_name}</strong><small>{prediction.path.length} steps</small></div>{prediction.path.map((step, index) => <article className="path-step" key={`${step.node_id}-${index}`}><span className="path-step-number">{index + 1}</span><div><strong>{step.type === "leaf" ? `Leaf prediction: ${step.prediction_name}` : `${step.feature} ${step.operator} ${Number(step.threshold).toPrecision(4)}`}</strong>{step.type === "split" && <p>Value {Number(step.value).toPrecision(4)} → follow the <b>{step.branch}</b> branch.</p>}</div></article>)}</div>}</section>
            )}

            {section === "dataset" && <DatasetReport result={result} />}
            {section === "hyperparameters" && <HyperparameterReport
              result={result} file={file} loading={loading} onTrain={train}
              maxDepth={maxDepth} setMaxDepth={setMaxDepth}
              criterion={criterion} setCriterion={setCriterion} splitter={splitter} setSplitter={setSplitter}
              minSamplesSplit={minSamplesSplit} setMinSamplesSplit={setMinSamplesSplit}
              minSamplesLeaf={minSamplesLeaf} setMinSamplesLeaf={setMinSamplesLeaf}
              maxLeafNodes={maxLeafNodes} setMaxLeafNodes={setMaxLeafNodes}
              maxFeatures={maxFeatures} setMaxFeatures={setMaxFeatures}
              classWeight={classWeight} setClassWeight={setClassWeight}
              minImpurityDecrease={minImpurityDecrease} setMinImpurityDecrease={setMinImpurityDecrease}
              ccpAlpha={ccpAlpha} setCcpAlpha={setCcpAlpha}
              features={plotFeatures}
            />}
            </>}
          </>
        )}
      </main>
    </div>
  );
}

function DataPreparation({ file, preview, target, setTarget, strategy, setStrategy, onChooseFile, onTrain, loading, existingResult }) {
  const missingByColumn = preview.headers.map((column) => ({
    column,
    missing: preview.rows.filter((row) => isMissingCsvCell(row[column])).length,
    numeric: preview.rows.map((row) => String(row[column] ?? "").trim()).filter((value) => !isMissingCsvCell(value)).every((value) => Number.isFinite(Number(value))),
  }));
  const missingCells = missingByColumn.reduce((sum, column) => sum + column.missing, 0);
  return (
    <section className="panel focused-panel preparation-page">
      <SectionHeading eyebrow="DATA QUALITY" title="Prepare your CSV" detail="Review missing values, choose how to fill feature columns, then train with the cleaned dataset." />
      <div className="preparation-upload-row">
        <div><strong>{file?.name || "No CSV selected"}</strong><p>{preview.rows.length ? `${preview.rows.length} rows · ${preview.headers.length} columns` : "Upload a CSV file to inspect its missing values."}</p></div>
        <button className="secondary-button" type="button" onClick={onChooseFile}>Choose CSV</button>
      </div>
      {preview.headers.length > 0 && <>
        <div className="preparation-summary-grid">
          <article><span>Missing cells</span><strong>{missingCells}</strong></article>
          <article><span>Columns with gaps</span><strong>{missingByColumn.filter((column) => column.missing > 0).length}</strong></article>
          <article><span>Rows without target</span><strong>{missingByColumn.find((column) => column.column === target)?.missing ?? 0}</strong></article>
        </div>
        <section className="preparation-section">
          <div className="panel-heading"><div><h2>Missing values by column</h2><p>Target rows without a class label are skipped during training.</p></div></div>
          <div className="table-scroll"><table className="report-table missing-table"><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Handling</th></tr></thead><tbody>{missingByColumn.map(({ column, missing, numeric }) => <tr key={column}><th>{column}</th><td>{numeric ? "Numeric" : "Category"}</td><td>{missing}</td><td>{missing === 0 ? "No changes" : column === target ? "Drop unlabeled rows" : numeric ? `${strategy} imputation` : "Mode imputation"}</td></tr>)}</tbody></table></div>
        </section>
        <section className="preparation-section">
          <div className="panel-heading"><div><h2>Fill missing feature values</h2><p>Numeric features use your chosen method. Categorical features use the most frequent value.</p></div></div>
          <div className="imputation-controls">
            <label>Numeric imputation method<select value={strategy} onChange={(event) => setStrategy(event.target.value)}><option value="mean">Mean</option><option value="median">Median</option><option value="mode">Mode</option></select></label>
            <label>Target column<select value={target} onChange={(event) => setTarget(event.target.value)}>{preview.headers.map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
          </div>
        </section>
        <section className="preparation-section preview-section">
          <div className="panel-heading"><div><h2>Data preview</h2><p>Blank cells are marked as missing. Showing up to 12 rows.</p></div></div>
          <div className="table-scroll"><table className="report-table dataset-table"><thead><tr>{preview.headers.map((column) => <th key={column}>{column}{column === target ? " · target" : ""}</th>)}</tr></thead><tbody>{preview.rows.slice(0, 12).map((row, rowIndex) => <tr key={rowIndex}>{preview.headers.map((column) => <td className={isMissingCsvCell(row[column]) ? "missing-cell" : ""} key={column}>{isMissingCsvCell(row[column]) ? "Missing" : row[column]}</td>)}</tr>)}</tbody></table></div>
        </section>
        {existingResult?.dataset?.imputation && <p className="imputation-result">Last training filled {existingResult.dataset.imputation.imputed_cells} feature cells with {existingResult.dataset.imputation.strategy} (categorical columns use mode).</p>}
        <div className="preparation-actions"><button className="primary-button" type="button" onClick={onTrain} disabled={!file || loading}>{loading ? "Preparing and training…" : "Apply and train tree"}</button></div>
      </>}
    </section>
  );
}

function NodeInformation({ node, classes, showTitle = true }) {
  if (!node) return null;
  return (
    <aside className="node-information">
      {showTitle && <h3>Node Information</h3>}
      <div className="node-rule"><span>Selected node · {node.id}</span><strong>{node.is_leaf ? node.prediction_name : `${node.feature} ≤ ${Number(node.threshold).toPrecision(4)}`}</strong></div>
      <dl><div><dt>Gini impurity</dt><dd>{Number(node.gini).toFixed(3)}</dd></div><div><dt>Samples</dt><dd>{node.samples}</dd></div><div><dt>Prediction</dt><dd>{node.prediction_name}</dd></div></dl>
      <h4>Class distribution</h4>
      {node.class_distribution.map((count, index) => <div className="distribution-row" key={classes[index]}><span>{classes[index]}</span><div><i style={{ width: `${node.samples ? (count / node.samples) * 100 : 0}%`, background: CLASS_COLORS[index % CLASS_COLORS.length] }} /></div><strong>{count}</strong></div>)}
    </aside>
  );
}

function PerformanceReport({ result }) {
  const metrics = result.metrics;
  const summary = [
    ["Accuracy", metrics.accuracy],
    ["Weighted precision", metrics.precision],
    ["Weighted recall", metrics.recall],
    ["Weighted F1", metrics.f1],
    ["Balanced accuracy", metrics.balanced_accuracy],
    ["Macro precision", metrics.macro_precision],
    ["Macro recall", metrics.macro_recall],
    ["Macro F1", metrics.macro_f1],
    ["Specificity", metrics.specificity],
    ["Negative predictive value", metrics.negative_predictive_value],
    ["False positive rate", metrics.false_positive_rate],
    ["ROC AUC (weighted)", metrics.roc_auc],
    ["Matthews correlation", metrics.matthews_correlation],
    ["Cohen's kappa", metrics.cohen_kappa],
    ["Generalization gap", result.complexity.generalization_gap],
  ];
  return (
    <>
      <section className="metric-detail-grid">{summary.map(([name, value]) => <article className="metric-detail-card" key={name}><span>{name}</span><strong>{value == null ? "n/a" : Number(value).toFixed(3)}</strong></article>)}</section>
      <section className="panel report-panel"><SectionHeading eyebrow="PER-CLASS REPORT" title="Classification Metrics" detail="One-vs-rest statistics for each target class." /><div className="table-scroll"><table className="report-table"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>NPV</th><th>FPR</th><th>ROC AUC</th><th>F1 score</th><th>Support</th></tr></thead><tbody>{metrics.per_class.map((item) => <tr key={item.class_name}><th>{item.class_name}</th><td>{percent(item.precision)}</td><td>{percent(item.recall)}</td><td>{percent(item.specificity)}</td><td>{percent(item.negative_predictive_value)}</td><td>{percent(item.false_positive_rate)}</td><td>{percent(item.roc_auc)}</td><td>{percent(item.f1)}</td><td>{item.support}</td></tr>)}</tbody></table></div></section>
    </>
  );
}

function DatasetReport({ result }) {
  return (
    <section className="panel focused-panel"><SectionHeading eyebrow="DATASET OVERVIEW" title={result.dataset.filename} detail={`${result.dataset.rows} rows · ${result.dataset.columns} columns · target: ${result.dataset.target}`} /><div className="feature-chip-list dataset-feature-list">{result.dataset.features.map((feature) => <span key={feature}>{feature}</span>)}</div><div className="table-scroll dataset-table-scroll"><table className="report-table dataset-table"><thead><tr><th>#</th>{result.dataset.features.map((feature) => <th key={feature}>{feature}</th>)}<th>{result.dataset.target}</th></tr></thead><tbody>{result.dataset.rows_data.slice(0, 100).map((row, index) => <tr key={index}><td>{index + 1}</td>{result.dataset.features.map((feature) => <td key={feature}>{String(row[feature])}</td>)}<td>{String(row[result.dataset.target])}</td></tr>)}</tbody></table></div>{result.dataset.rows > 100 && <p className="table-footnote">Showing the first 100 of {result.dataset.rows} rows.</p>}</section>
  );
}

function TreeControls({
  result, file, loading, onTrain, maxDepth, setMaxDepth,
  criterion, setCriterion, splitter, setSplitter,
  minSamplesSplit, setMinSamplesSplit, minSamplesLeaf, setMinSamplesLeaf,
  maxLeafNodes, setMaxLeafNodes, maxFeatures, setMaxFeatures,
  classWeight, setClassWeight, minImpurityDecrease, setMinImpurityDecrease,
  ccpAlpha, setCcpAlpha, features = [],
}) {
  const sampleLimit = Math.max(2, Math.min(100, result.dataset.train_rows));
  const changed = maxDepth !== result.complexity.requested_max_depth
    || criterion !== result.complexity.requested_criterion
    || splitter !== result.complexity.requested_splitter
    || minSamplesSplit !== result.complexity.requested_min_samples_split
    || minSamplesLeaf !== result.complexity.requested_min_samples_leaf
    || maxLeafNodes !== result.complexity.requested_max_leaf_nodes
    || maxFeatures !== result.complexity.requested_max_features
    || classWeight !== result.complexity.requested_class_weight
    || minImpurityDecrease !== result.complexity.requested_min_impurity_decrease
    || ccpAlpha !== result.complexity.requested_ccp_alpha;
  const ranges = [
    ["Maximum tree depth", "max-depth-range", maxDepth, 1, 20, 1, setMaxDepth, String(maxDepth)],
    ["Minimum samples to split", "min-split-range", minSamplesSplit, 2, Math.max(sampleLimit, minSamplesSplit), 1, setMinSamplesSplit, String(minSamplesSplit)],
    ["Minimum samples per leaf", "min-leaf-range", minSamplesLeaf, 1, Math.max(sampleLimit, minSamplesLeaf), 1, setMinSamplesLeaf, String(minSamplesLeaf)],
    ["Maximum leaf nodes", "max-leaf-range", maxLeafNodes, 0, Math.max(sampleLimit, maxLeafNodes), 1, setMaxLeafNodes, maxLeafNodes === 0 ? "Unlimited" : String(maxLeafNodes)],
    ["Minimum impurity decrease", "min-impurity-range", minImpurityDecrease, 0, Math.max(0.2, minImpurityDecrease), 0.001, setMinImpurityDecrease, Number(minImpurityDecrease).toFixed(3)],
    ["Pruning alpha", "pruning-alpha-range", ccpAlpha, 0, Math.max(0.1, ccpAlpha), 0.001, setCcpAlpha, Number(ccpAlpha).toFixed(3)],
  ];
  return (
    <section className="panel rail-panel tree-tuning-panel">
      <div className="tree-tuning-heading"><div><h3>Tree Controls</h3><p>Adjust values with the sliders or arrow keys, then retrain.</p></div><span className="actual-depth-badge">Actual depth {result.complexity.actual_depth}</span></div>
      <div className="tree-range-list">
        {ranges.map(([label, id, value, min, max, step, update, display]) => (
          <label className="tree-range-control" htmlFor={id} key={id}>
            <span className="tree-range-label"><span>{label}</span><strong>{display}</strong></span>
            <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(event) => update(Number(event.target.value))} />
            <span className="tree-range-ends"><small>{min === 0 && id === "max-leaf-range" ? "Unlimited" : min}</small><small>{max}</small></span>
          </label>
        ))}
      </div>
      <details className="tree-advanced-controls" open>
        <summary>More parameters</summary>
        <label>Split criterion<select value={criterion} onChange={(event) => setCriterion(event.target.value)}><option value="gini">Gini</option><option value="entropy">Entropy</option><option value="log_loss">Log loss</option></select></label>
        <label>Split strategy<select value={splitter} onChange={(event) => setSplitter(event.target.value)}><option value="best">Best</option><option value="random">Random</option></select></label>
        <label>Features per split<select value={maxFeatures} onChange={(event) => setMaxFeatures(event.target.value)}><option value="all">All features</option><option value="sqrt">Square root</option><option value="log2">Log₂</option></select></label>
        <label>Class weighting<select value={classWeight} onChange={(event) => setClassWeight(event.target.value)}><option value="none">None</option><option value="balanced">Balanced</option></select></label>
      </details>
      <button className="primary-button tree-apply-button" type="button" onClick={() => onTrain()} disabled={!file || loading}>{loading ? "Training…" : changed ? "Apply settings & retrain" : "Retrain model"}</button>
      {features.length > 0 && <p className="tree-control-features">Selected features: {features.slice(0, 3).join(", ")}</p>}
    </section>
  );
}

function HyperparameterReport({ result, ...controlProps }) {
  const config = result.complexity;
  const items = [
    ["Maximum depth (configured)", config.requested_max_depth], ["Tree depth (actual)", config.actual_depth],
    ["Split criterion", config.requested_criterion], ["Split strategy", config.requested_splitter],
    ["Minimum samples to split", config.requested_min_samples_split], ["Minimum samples per leaf", config.requested_min_samples_leaf],
    ["Maximum leaf nodes", config.requested_max_leaf_nodes || "Unlimited"], ["Features per split", config.requested_max_features],
    ["Class weighting", config.requested_class_weight], ["Minimum impurity decrease", config.requested_min_impurity_decrease],
    ["Cost-complexity pruning alpha", config.requested_ccp_alpha], ["Fixed random seed", config.random_state],
    ["Node count", config.node_count], ["Leaf count", config.leaf_count],
    ["Training accuracy", percent(config.train_accuracy)], ["Validation accuracy", percent(config.validation_accuracy)],
  ];
  const descriptions = {
    "Maximum depth (configured)": "Limits the number of sequential decisions along a path.",
    "Split criterion": "Scores class purity after a split; Gini is the default, entropy and log loss use information gain.",
    "Split strategy": "Best selects the strongest candidate; random samples a candidate threshold.",
    "Minimum samples to split": "A node needs at least this many rows before it can create children.",
    "Minimum samples per leaf": "Keeps each terminal group from becoming too small and fragile.",
    "Maximum leaf nodes": "Caps the number of terminal rules; unlimited leaves are allowed when set to zero.",
    "Features per split": "Restricts the candidate feature set. Fewer features can add variety but may miss a strong split.",
    "Class weighting": "Balanced weighting gives greater influence to underrepresented classes.",
    "Minimum impurity decrease": "Requires a split to improve purity by at least this amount.",
    "Cost-complexity pruning alpha": "A larger alpha penalizes complexity and prunes more branches.",
  };
  return <section className="panel focused-panel"><SectionHeading eyebrow="TREE CONTROLS" title="Hyperparameters" detail="Customize numeric settings with sliders and fine-tune split options. Apply changes to train a new tree." /><TreeControls result={result} {...controlProps} /><div className="hyperparameter-grid">{items.map(([name, value]) => <article className="hyperparameter-card" key={name}><span>{name}</span><strong>{value ?? "—"}</strong>{descriptions[name] && <small>{descriptions[name]}</small>}</article>)}</div><p className="table-footnote">The random seed is fixed at 42 for reproducible comparisons. Validation accuracy is a held-out estimate, not a guarantee of future performance.</p></section>;
}

function DecisionTreeGuide({ maxDepth, setMaxDepth, result, file, loading, onTrain }) {
  const requestedDepth = result?.complexity?.requested_max_depth;
  const depthPending = result && Number(requestedDepth) !== Number(maxDepth);
  const depthDescription = maxDepth <= 2
    ? "A shallow tree is easier to explain and less likely to memorize noise, but it may miss useful patterns."
    : maxDepth <= 5
      ? "This is a moderate depth: enough room for several useful rules while keeping the tree reasonably readable."
      : "A deeper tree can capture fine-grained patterns, but it can become hard to read and may overfit the training data.";
  return (
    <section className="panel focused-panel guide-page">
      <SectionHeading eyebrow="LEARN · EXPLORE · APPLY" title="Decision Tree Guide" detail="A practical, detailed guide to the model, the controls in this workspace, and how to read every visualization." />
      <div className="guide-intro">
        <div className="guide-intro-icon" aria-hidden="true">⑂</div>
        <div><h3>What is a decision tree?</h3><p>A decision tree is a sequence of if–then questions. Starting at the root, it checks one feature at a time, follows a branch for the answer, and ends at a leaf that gives a prediction. The same structure can be read as a flowchart or as a set of human-readable rules.</p></div>
      </div>

      <section className="guide-depth-lab">
        <div className="guide-depth-heading"><div><span className="section-eyebrow">INTERACTIVE DEPTH LAB</span><h3>What changes when tree depth increases?</h3><p>Choose a maximum depth, then train/update the tree to see the actual structure and validation results change.</p></div><strong className="depth-value">{maxDepth}<small> max levels</small></strong></div>
        <label className="guide-depth-slider" htmlFor="guide-max-depth"><span>Shallower · simpler</span><input id="guide-max-depth" type="range" min="1" max="20" value={maxDepth} onChange={(event) => setMaxDepth(Number(event.target.value))} /><span>Deeper · more detailed</span></label>
        <p className="guide-depth-explanation">{depthDescription}</p>
        <div className="guide-depth-footer">
          <span>{result ? `Current trained tree: depth ${result.complexity.actual_depth}, ${result.complexity.node_count} nodes, ${result.complexity.leaf_count} leaves.` : "Train a model to see its measured depth, node count, and leaf count here."}</span>
          <button className="primary-button" type="button" onClick={() => onTrain()} disabled={!file || loading}>{loading ? "Training…" : depthPending ? `Apply depth ${maxDepth} and retrain` : result ? "Update visualization" : "Train decision tree"}</button>
        </div>
        {depthPending && <p className="guide-pending-note">The selected depth is a pending setting. The displayed tree and scores still belong to the last trained model until you update it.</p>}
      </section>

      {result && <div className="guide-model-snapshot">
        <article><span>Training accuracy</span><strong>{(result.complexity.train_accuracy * 100).toFixed(1)}%</strong></article>
        <article><span>Validation accuracy</span><strong>{(result.complexity.validation_accuracy * 100).toFixed(1)}%</strong></article>
        <article><span>Generalization gap</span><strong>{(result.complexity.generalization_gap * 100).toFixed(1)}%</strong></article>
        <article><span>Current leaves</span><strong>{result.complexity.leaf_count}</strong></article>
      </div>}

      <div className="guide-section-grid">
        <article className="guide-card">
          <span className="guide-card-number">01</span><h3>Read the tree from top to bottom</h3>
          <p><b>Root:</b> the first question, applied to every row. <b>Split node:</b> a feature test that divides the rows. <b>Branch:</b> the outcome of that test—True/False for numeric thresholds. <b>Leaf:</b> the final prediction for rows that arrive there.</p>
          <p>Each node also reports sample count and impurity. A node with many samples represents a broad portion of the data; a leaf with very few samples may describe a narrow, fragile rule. Select a node to inspect its class distribution and predicted class in the information card.</p>
        </article>
        <article className="guide-card">
          <span className="guide-card-number">02</span><h3>How a split is chosen</h3>
          <p>During training, the algorithm considers candidate features and thresholds. It chooses the split that most reduces the mixedness of the child groups, subject to the configured constraints. This is a greedy process: each split is chosen using the rows currently at that node.</p>
          <p>For classification with Gini impurity, <code>Gini = 1 − Σ(pᵢ²)</code>, where <code>pᵢ</code> is the fraction of rows in class <code>i</code>. Gini is 0 when a node contains only one class; it is larger when classes are more evenly mixed. The split score is the parent impurity minus the sample-weighted impurity of its children.</p>
          <p>For regression trees, the same idea is applied to numeric targets, typically by reducing the within-node squared error/variance rather than class impurity.</p>
        </article>
        <article className="guide-card">
          <span className="guide-card-number">03</span><h3>How a prediction is made</h3>
          <p>A row starts at the root. For a rule such as <code>Study_Hours ≤ 3.85</code>, the row’s value is compared with 3.85. It follows the True or False edge and repeats until it reaches a leaf. In classification, the leaf predicts its most common training class; in regression, it predicts a numeric summary of the leaf’s training targets.</p>
          <p>Use <b>Prediction</b> in the navigation to trace a particular dataset row, see each comparison, and inspect the final result. A model prediction is not a causal explanation: the path describes how this fitted tree reached its answer, not why the feature caused that outcome.</p>
        </article>
        <article className="guide-card">
          <span className="guide-card-number">04</span><h3>Depth, overfitting, and generalization</h3>
          <p>Increasing maximum depth allows more sequential questions. It can reduce training error because the model can form narrower regions and more specific leaves. But a very deep tree may learn quirks or noise in the training sample and perform worse on new rows.</p>
          <p>Compare training and held-out validation scores, not just training accuracy. A high training score with a noticeably lower validation score is a warning sign of overfitting. A shallow tree with both scores low may be underfitting. Validation results vary with the data split, so compare changes on the same dataset and treat small differences cautiously.</p>
          <p>Depth is not the only control: <b>minimum samples per leaf</b> prevents tiny leaves; <b>pruning alpha</b> penalizes a tree for extra complexity. These controls trade detail against stability and readability.</p>
        </article>
        <article className="guide-card">
          <span className="guide-card-number">05</span><h3>Use each visualization for a different question</h3>
          <ul>
            <li><b>Overview:</b> a quick model summary and compact comparison of boundary, importance, confusion matrix, and learning curve.</li>
            <li><b>3D Tree:</b> orbit the fitted tree, zoom, and double-click nodes for inspection.</li>
            <li><b>3D Surface:</b> explore predicted regions in three selected numeric features; observed samples are shown over the regions.</li>
            <li><b>3D Scatter:</b> examine the same selected feature axes, colored by the observed target class.</li>
            <li><b>2D Decision Boundary:</b> see predicted classes across two features while points show observed rows. Other features are held at representative values for the projection.</li>
            <li><b>Feature Importance:</b> compare each feature’s accumulated contribution to impurity reduction. It is model-specific and does not prove causation.</li>
            <li><b>Confusion Matrix:</b> rows are actual classes and columns are predictions; diagonal cells are correct, off-diagonal cells show which classes are confused.</li>
            <li><b>Learning Curve / Performance:</b> review training-vs-validation behavior and class-level metrics. Scores describe the evaluation split, not a guarantee of future performance.</li>
          </ul>
        </article>
        <article className="guide-card">
          <span className="guide-card-number">06</span><h3>A reliable workflow in this app</h3>
          <ol>
            <li>Open <b>Data Preparation</b>, upload a CSV, confirm the target, and inspect missing cells. Choose mean, median, or mode imputation for numeric feature gaps; categorical gaps use mode.</li>
            <li>Set tree depth, minimum samples per leaf, and pruning alpha in the left panel. Select the target column and ensure it is not being used as a feature.</li>
            <li>Choose <b>Update Visualization</b> to fit the model. The page’s tree and scores update only after training finishes.</li>
            <li>Start with a shallow tree and inspect its rules. Increase depth in small increments, retrain, then compare validation performance and tree size.</li>
            <li>Use the confusion matrix to identify class-specific errors and the prediction trace to understand individual row paths.</li>
            <li>Check that the evaluation is appropriate for the problem. Avoid leakage from future information, IDs, or features that encode the answer.</li>
          </ol>
        </article>
        <article className="guide-card guide-card-wide">
          <span className="guide-card-number">07</span><h3>Strengths, limitations, and interpretation cautions</h3>
          <p><b>Strengths:</b> a single tree can express nonlinear thresholds and feature interactions, needs little feature scaling, and can be explained as a path of rules. It handles mixed decision logic naturally and provides a useful baseline.</p>
          <p><b>Limitations:</b> small data changes can produce a different tree; greedy splits do not guarantee a globally optimal tree; deep trees can overfit; and one tree may be less accurate than an ensemble. Feature importance can favor features with many possible split points and should be interpreted alongside domain knowledge and validation results.</p>
          <p><b>Before trusting a result:</b> verify the target and feature columns, check class balance and missing-value handling, use a representative holdout or cross-validation, compare relevant metrics, and keep the model’s intended use and costs of errors in mind. This visualizer is an educational and exploratory aid, not a substitute for domain review.</p>
        </article>
      </div>
    </section>
  );
}

export default App;
