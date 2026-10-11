import P3Lab from "./P3Lab.jsx";
import { useEffect, useMemo, useRef, useState } from "react";

import "./App.css";

function buildLayout(
  nodes,
  nodeId = 0,
  depth = 0,
  positions = {},
  leafIndex = { value: 0 },
) {
  const node = nodes.find((item) => item.id === nodeId);

  if (!node) {
    return positions;
  }

  const leftExists =
    node.left !== -1 && nodes.some((item) => item.id === node.left);

  const rightExists =
    node.right !== -1 && nodes.some((item) => item.id === node.right);

  if (!leftExists && !rightExists) {
    positions[nodeId] = {
      x: leafIndex.value * 240,
      y: depth * 154,
    };

    leafIndex.value += 1;

    return positions;
  }

  if (leftExists) {
    buildLayout(nodes, node.left, depth + 1, positions, leafIndex);
  }

  if (rightExists) {
    buildLayout(nodes, node.right, depth + 1, positions, leafIndex);
  }

  const children = [];

  if (leftExists) {
    children.push(positions[node.left]);
  }

  if (rightExists) {
    children.push(positions[node.right]);
  }

  const averageX =
    children.reduce((sum, child) => sum + child.x, 0) / children.length;

  positions[nodeId] = {
    x: averageX,
    y: depth * 154,
  };

  return positions;
}

function DecisionTree({
  nodes,
  selectedNode,
  onSelect,
  predictionPath,
  linkedFeature,
}) {
  const layout = useMemo(() => buildLayout(nodes), [nodes]);
  const treeViewportRef = useRef(null);
  const [viewportWidth, setViewportWidth] = useState(0);
  const [manualZoom, setManualZoom] = useState(null);

  useEffect(() => {
    const viewport = treeViewportRef.current;

    if (!viewport || typeof ResizeObserver === "undefined") {
      return;
    }

    const observer = new ResizeObserver(([entry]) => {
      setViewportWidth(entry.contentRect.width);
    });

    observer.observe(viewport);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setManualZoom(null);
  }, [nodes]);

  const predictionPathNodes = useMemo(
    () => new Set((predictionPath?.path || []).map((step) => step.node_id)),
    [predictionPath],
  );

  const predictionPathEdges = useMemo(
    () =>
      new Set(
        (predictionPath?.path || [])
          .filter((step) => step.next_node !== undefined)
          .map((step) => `${step.node_id}-${step.next_node}`),
      ),
    [predictionPath],
  );

  if (!nodes.length) {
    return <div className="empty-tree">No tree available.</div>;
  }

  const maxX = Math.max(
    ...Object.values(layout).map((position) => position.x),
    0,
  );

  const maxY = Math.max(
    ...Object.values(layout).map((position) => position.y),
    0,
  );

  const contentWidth = maxX + 200;
  const width = Math.max(720, contentWidth + 80);
  const height = Math.max(380, maxY + 150);
  const offsetX = (width - contentWidth) / 2;
  const fitZoom = viewportWidth
    ? Math.min(1, Math.max(0.05, (viewportWidth - 48) / width))
    : 1;
  const zoom = manualZoom ?? fitZoom;

  const adjustZoom = (amount) => {
    setManualZoom((current) =>
      Math.min(2.5, Math.max(0.1, (current ?? fitZoom) + amount)),
    );
  };

  return (
    <div className="tree-visual">
      <div className="tree-zoom-controls" aria-label="Tree zoom controls">
        <button
          type="button"
          onClick={() => adjustZoom(-0.15)}
          aria-label="Zoom out"
        >
          −
        </button>
        <output aria-live="polite">{Math.round(zoom * 100)}%</output>
        <button
          type="button"
          onClick={() => adjustZoom(0.15)}
          aria-label="Zoom in"
        >
          +
        </button>
        <button type="button" onClick={() => setManualZoom(null)}>
          Fit
        </button>
      </div>

      <div className="tree-scroll" ref={treeViewportRef}>
        <svg
          className="tree-svg"
          width={Math.round(width * zoom)}
          height={Math.round(height * zoom)}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Interactive decision tree diagram"
        >
          <g transform={`translate(${offsetX}, 0)`}>
            {/* CONNECTIONS */}

            {nodes.map((node) => {
              if (node.is_leaf) {
                return null;
              }

              const parent = layout[node.id];

              if (!parent) {
                return null;
              }

              const branches = [
                {
                  id: node.left,
                  label: "YES",
                },
                {
                  id: node.right,
                  label: "NO",
                },
              ];

              return branches.map((branch) => {
                if (branch.id === -1 || !layout[branch.id]) {
                  return null;
                }

                const child = layout[branch.id];

                const edgeKey = `${node.id}-${branch.id}`;

                const pathActive = predictionPathEdges.has(edgeKey);

                const pathMode = Boolean(predictionPath);

                const startX = parent.x + 100;

                const startY = parent.y + 108;

                const endX = child.x + 100;

                const endY = child.y;

                const middleY = (startY + endY) / 2;

                return (
                  <g key={`${node.id}-${branch.id}`}>
                    <path
                      id={`prediction-edge-${edgeKey}`}
                      d={`
                      M ${startX} ${startY}
                      C ${startX} ${middleY},
                        ${endX} ${middleY},
                        ${endX} ${endY}
                    `}
                      fill="none"
                      stroke={
                        pathActive
                          ? "#4f46e5"
                          : pathMode
                            ? "#e2e8f0"
                            : "#cbd5e1"
                      }
                      strokeWidth={pathActive ? 4 : 2}
                      opacity={pathMode && !pathActive ? 0.45 : 1}
                    />

                    <rect
                      x={(startX + endX) / 2 - 20}
                      y={middleY - 11}
                      width="40"
                      height="22"
                      rx="11"
                      fill="white"
                      opacity={pathMode && !pathActive ? 0.55 : 1}
                    />

                    <text
                      x={(startX + endX) / 2}
                      y={middleY + 4}
                      textAnchor="middle"
                      fontSize="10"
                      fontWeight="700"
                      fill="#64748b"
                    >
                      {branch.label}
                    </text>
                  </g>
                );
              });
            })}

            {/* ANIMATED ROW FLOW */}

            {predictionPath &&
              predictionPath.path
                .filter(
                  (step) =>
                    step.type === "split" && step.next_node !== undefined,
                )
                .map((step, index) => {
                  const edgeKey = `${step.node_id}-${step.next_node}`;

                  return (
                    <g
                      key={`row-flow-${edgeKey}-${index}`}
                      className="row-flow-marker"
                      pointerEvents="none"
                    >
                      <circle
                        r="8"
                        fill="#4f46e5"
                        stroke="#ffffff"
                        strokeWidth="3"
                        opacity="0"
                      >
                        <animateMotion
                          dur="1.1s"
                          begin={`${index * 1.1}s`}
                          fill="remove"
                          repeatCount="1"
                        >
                          <mpath href={`#prediction-edge-${edgeKey}`} />
                        </animateMotion>

                        <animate
                          attributeName="opacity"
                          values="0;1;1;0"
                          keyTimes="0;0.02;0.9;1"
                          dur="1.1s"
                          begin={`${index * 1.1}s`}
                          fill="remove"
                        />
                      </circle>

                      {index === 0 && (
                        <text
                          x="0"
                          y="-14"
                          textAnchor="middle"
                          fontSize="10"
                          fontWeight="800"
                          fill="#4f46e5"
                        >
                          ROW
                        </text>
                      )}
                    </g>
                  );
                })}

            {/* TREE NODES */}

            {nodes.map((node) => {
              const position = layout[node.id];

              if (!position) {
                return null;
              }

              const selected = selectedNode?.id === node.id;

              const featureLinked =
                Boolean(linkedFeature) && node.feature === linkedFeature;
              const pathActive = predictionPathNodes.has(node.id);

              const pathMode = Boolean(predictionPath);

              return (
                <g
                  key={node.id}
                  transform={`
                translate(
                  ${position.x},
                  ${position.y}
                )
              `}
                  onClick={() => onSelect(node)}
                  role="button"
                  tabIndex="0"
                  aria-label={
                    node.is_leaf
                      ? `Leaf node ${node.id}: predicts ${node.prediction_name}`
                      : `Decision node ${node.id}: ${node.feature} less than or equal to ${node.threshold}`
                  }
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSelect(node);
                    }
                  }}
                  className="tree-node-group"
                  opacity={
                    linkedFeature && !featureLinked
                      ? 0.28
                      : pathMode && !pathActive
                        ? 0.38
                        : 1
                  }
                >
                  <rect
                    width="200"
                    height="108"
                    rx="14"
                    fill={selected ? "#eef2ff" : "#ffffff"}
                    stroke={
                      featureLinked
                        ? "#f59e0b"
                        : selected || pathActive
                          ? "#4f46e5"
                          : node.is_leaf
                            ? "#10b981"
                            : "#dbe1ea"
                    }
                    strokeWidth={
                      featureLinked ? 4 : selected ? 4 : pathActive ? 3 : 2
                    }
                  />

                  {node.is_leaf ? (
                    <>
                      <text
                        x="100"
                        y="27"
                        textAnchor="middle"
                        fontSize="11"
                        fontWeight="800"
                        fill="#059669"
                      >
                        LEAF NODE
                      </text>

                      <text
                        x="100"
                        y="51"
                        textAnchor="middle"
                        fontSize="14"
                        fontWeight="800"
                        fill="#111827"
                      >
                        {node.prediction_name}
                      </text>

                      <text
                        x="100"
                        y="73"
                        textAnchor="middle"
                        fontSize="11"
                        fill="#64748b"
                      >
                        Samples: {node.samples}
                      </text>

                      <text
                        x="100"
                        y="91"
                        textAnchor="middle"
                        fontSize="10"
                        fill="#94a3b8"
                      >
                        Gini: {node.gini}
                      </text>
                    </>
                  ) : (
                    <>
                      <text
                        x="100"
                        y="22"
                        textAnchor="middle"
                        fontSize="10"
                        fontWeight="900"
                        fill="#4f46e5"
                      >
                        DECISION
                      </text>

                      <text
                        x="100"
                        y="45"
                        textAnchor="middle"
                        fontSize="13"
                        fontWeight="800"
                        fill="#111827"
                      >
                        {node.feature}
                      </text>

                      <text
                        x="100"
                        y="66"
                        textAnchor="middle"
                        fontSize="12"
                        fill="#475569"
                      >
                        ≤ {node.threshold?.toFixed(3)}
                      </text>

                      <text
                        x="100"
                        y="87"
                        textAnchor="middle"
                        fontSize="10"
                        fill="#64748b"
                      >
                        Gini: {node.gini}
                        {" • "}
                        Samples: {node.samples}
                      </text>
                    </>
                  )}
                </g>
              );
            })}
          </g>
        </svg>
      </div>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <span>{label}</span>

      <strong>{(value * 100).toFixed(1)}%</strong>

      <div className="metric-track">
        <div
          className="metric-fill"
          style={{
            width: `${value * 100}%`,
          }}
        />
      </div>
    </div>
  );
}

function DynamicBuildPanel({ tree = [], dataset = null }) {
  const [step, setStep] = React.useState(0);

  const steps = [
    {
      title: "1. Load Dataset",
      description: dataset
        ? `ModelMind loaded ${dataset.rows} rows and ${dataset.columns} columns.`
        : "Upload a dataset to begin.",
    },
    {
      title: "2. Find the Best Split",
      description: tree.length
        ? `The model searches for the feature and threshold that best separates the classes.`
        : "The model will calculate possible splits.",
    },
    {
      title: "3. Create Root Node",
      description: tree.length
        ? `Root split: ${tree[0]?.feature ?? "calculating"} ${tree[0]?.threshold != null ? `≤ ${Number(tree[0].threshold).toFixed(2)}` : ""}`
        : "The root node will appear after training.",
    },
    {
      title: "4. Build Branches",
      description: tree.length
        ? `The tree continues splitting the remaining samples into smaller groups.`
        : "Branches will be created from the selected split.",
    },
    {
      title: "5. Final Model",
      description: tree.length
        ? `Decision Tree created with ${tree.length} nodes.`
        : "Train the model to create the final tree.",
    },
  ];

  React.useEffect(() => {
    if (!tree.length) {
      setStep(0);
      return;
    }

    const timer = setInterval(() => {
      setStep((current) => {
        if (current >= steps.length - 1) {
          clearInterval(timer);
          return current;
        }
        return current + 1;
      });
    }, 1200);

    return () => clearInterval(timer);
  }, [tree.length]);

  return (
    <section className="dynamic-build-panel">
      <div className="dynamic-build-header">
        <div>
          <span className="dynamic-build-kicker">MODEL FORMATION</span>
          <h2>How ModelMind Builds the Tree</h2>
          <p>
            Watch the decision tree form step-by-step from the uploaded data.
          </p>
        </div>

        <div className="dynamic-build-progress">
          {steps.map((_, index) => (
            <span key={index} className={index <= step ? "active" : ""} />
          ))}
        </div>
      </div>

      <div className="dynamic-build-steps">
        {steps.map((item, index) => (
          <button
            key={item.title}
            className={`dynamic-build-step ${
              index === step ? "current" : ""
            } ${index < step ? "completed" : ""}`}
            onClick={() => setStep(index)}
          >
            <span className="dynamic-step-number">
              {index < step ? "✓" : index + 1}
            </span>

            <span>
              <strong>{item.title}</strong>
              <small>{item.description}</small>
            </span>
          </button>
        ))}
      </div>

      <div className="dynamic-build-visual">
        <div className={`build-node stage-${step}`}>
          <span className="build-node-label">
            {step === 0 && "DATASET"}
            {step === 1 && "BEST SPLIT"}
            {step === 2 && "ROOT NODE"}
            {step === 3 && "BRANCHES"}
            {step === 4 && "MODEL READY"}
          </span>

          {step === 0 && (
            <div className="build-data-animation">
              <span>sepal_length</span>
              <span>sepal_width</span>
              <span>petal_length</span>
              <span>petal_width</span>
              <span className="target">target</span>
            </div>
          )}

          {step === 1 && (
            <div className="build-search-animation">
              <div className="search-ring" />
              <strong>Finding optimal split...</strong>
              <small>Comparing features and thresholds</small>
            </div>
          )}

          {step === 2 && tree[0] && (
            <div className="build-rule">
              <strong>{tree[0].feature}</strong>
              <span>≤</span>
              <strong>{Number(tree[0].threshold).toFixed(2)}</strong>
            </div>
          )}

          {step === 3 && (
            <div className="mini-tree">
              <div className="mini-root">{tree[0]?.feature ?? "Feature"}</div>
              <div className="mini-lines">
                <span />
                <span />
              </div>
              <div className="mini-leaves">
                <b>YES</b>
                <b>NO</b>
              </div>
            </div>
          )}

          {step === 4 && (
            <div className="model-ready">
              <div className="ready-check">✓</div>
              <strong>Decision Tree Ready</strong>
              <small>{tree.length} nodes generated</small>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function SplitExplorer({ splitExplorer, selectedNode, onSelectNode }) {
  const nodes = Object.values(splitExplorer || {});

  const initialNode =
    selectedNode && splitExplorer?.[String(selectedNode.id)]
      ? String(selectedNode.id)
      : nodes.length
        ? String(nodes[0].node_id)
        : "";

  const [activeNodeId, setActiveNodeId] = useState(initialNode);

  const [candidateIndex, setCandidateIndex] = useState(0);

  const activeNode = splitExplorer?.[String(activeNodeId)] || nodes[0];

  if (!activeNode) {
    return null;
  }

  const candidates = activeNode.candidates || [];

  const selectedCandidate =
    candidates[Math.min(candidateIndex, Math.max(candidates.length - 1, 0))];

  const currentCandidate = candidates.find(
    (candidate) =>
      Math.abs(candidate.threshold - activeNode.current_threshold) < 0.00001,
  ) || {
    threshold: activeNode.current_threshold,
    weighted_gini: activeNode.current_gini,
    left_count: 0,
    right_count: 0,
  };

  const selectedIsCurrent =
    selectedCandidate &&
    Math.abs(selectedCandidate.threshold - activeNode.current_threshold) <
      0.00001;

  const giniChange =
    selectedCandidate && currentCandidate
      ? Number(currentCandidate.weighted_gini - selectedCandidate.weighted_gini)
      : 0;

  const leftChange =
    selectedCandidate && currentCandidate
      ? selectedCandidate.left_count - currentCandidate.left_count
      : 0;

  const rightChange =
    selectedCandidate && currentCandidate
      ? selectedCandidate.right_count - currentCandidate.right_count
      : 0;

  const chooseNode = (nodeId) => {
    setActiveNodeId(String(nodeId));
    setCandidateIndex(0);
  };

  return (
    <section className="section split-explorer-section">
      <div className="section-title">
        <span>EXPLAINABILITY</span>

        <h3>Split Explorer</h3>

        <p>
          Explore candidate thresholds and see how each split changes the child
          groups and weighted Gini impurity.
        </p>
      </div>

      <div className="split-explorer-card">
        <div className="split-explorer-header">
          <div>
            <label>Decision Node</label>

            <select
              value={activeNodeId}
              onChange={(event) => chooseNode(event.target.value)}
            >
              {nodes.map((node) => (
                <option key={node.node_id} value={node.node_id}>
                  Node {node.node_id} — {node.feature}
                </option>
              ))}
            </select>
          </div>

          <div className="split-stat">
            <span>FEATURE</span>

            <strong>{activeNode.feature}</strong>
          </div>

          <div className="split-stat">
            <span>CURRENT THRESHOLD</span>

            <strong>{Number(activeNode.current_threshold).toFixed(2)}</strong>
          </div>

          <div className="split-stat">
            <span>CURRENT GINI</span>

            <strong>{Number(activeNode.current_gini).toFixed(4)}</strong>
          </div>

          <div className="split-stat">
            <span>ROWS</span>

            <strong>{activeNode.samples}</strong>
          </div>
        </div>

        <div className="split-explorer-body">
          <div className="split-candidates">
            <div className="split-heading">
              <strong>Candidate Thresholds</strong>

              <span>Best score first</span>
            </div>

            {candidates.map((candidate, index) => {
              const selected = index === candidateIndex;

              const current =
                Math.abs(candidate.threshold - activeNode.current_threshold) <
                0.00001;

              return (
                <button
                  key={`${candidate.threshold}-${index}`}
                  type="button"
                  className={`split-candidate ${selected ? "active" : ""}`}
                  onClick={() => setCandidateIndex(index)}
                >
                  <div className="split-candidate-title">
                    <strong>≤ {Number(candidate.threshold).toFixed(2)}</strong>

                    {current && <span>CURRENT</span>}
                  </div>

                  <div className="split-candidate-gini">
                    <div>
                      <small>Weighted Gini</small>

                      <b>{Number(candidate.weighted_gini).toFixed(4)}</b>
                    </div>

                    <div>
                      <small>Left</small>

                      <b>{candidate.left_count}</b>
                    </div>

                    <div>
                      <small>Right</small>

                      <b>{candidate.right_count}</b>
                    </div>
                  </div>

                  {!current && (
                    <div className="split-candidate-action">
                      Click to compare with current split
                    </div>
                  )}
                </button>
              );
            })}
          </div>

          <div className="split-result-area">
            {selectedCandidate && (
              <div className="split-result">
                <div className="split-result-top">
                  <div>
                    <span>SELECTED SPLIT</span>

                    <h4>
                      {activeNode.feature} ≤{" "}
                      {Number(selectedCandidate.threshold).toFixed(2)}
                    </h4>
                  </div>

                  <div className="split-score">
                    <span>Weighted Gini</span>

                    <strong>
                      {Number(selectedCandidate.weighted_gini).toFixed(4)}
                    </strong>
                  </div>
                </div>

                <div className="split-children">
                  <div className="split-child left">
                    <div className="split-child-header">
                      <span>LEFT CHILD</span>

                      <strong>{selectedCandidate.left_count} rows</strong>
                    </div>

                    {selectedCandidate.left_distribution.map((count, index) => (
                      <div className="split-class" key={`left-${index}`}>
                        <span>Class {index}</span>

                        <div>
                          <span
                            style={{
                              width: `${
                                selectedCandidate.left_count
                                  ? (count / selectedCandidate.left_count) * 100
                                  : 0
                              }%`,
                            }}
                          />
                        </div>

                        <b>{count}</b>
                      </div>
                    ))}
                  </div>

                  <div className="split-child right">
                    <div className="split-child-header">
                      <span>RIGHT CHILD</span>

                      <strong>{selectedCandidate.right_count} rows</strong>
                    </div>

                    {selectedCandidate.right_distribution.map(
                      (count, index) => (
                        <div className="split-class" key={`right-${index}`}>
                          <span>Class {index}</span>

                          <div>
                            <span
                              style={{
                                width: `${
                                  selectedCandidate.right_count
                                    ? (count / selectedCandidate.right_count) *
                                      100
                                    : 0
                                }%`,
                              }}
                            />
                          </div>

                          <b>{count}</b>
                        </div>
                      ),
                    )}
                  </div>
                </div>

                <div className="split-explanation">
                  <strong>Why does this split matter?</strong>

                  <p>
                    This threshold sends <b>{selectedCandidate.left_count}</b>{" "}
                    rows to the left and <b>{selectedCandidate.right_count}</b>{" "}
                    rows to the right. The resulting weighted Gini is{" "}
                    <b>{Number(selectedCandidate.weighted_gini).toFixed(4)}</b>.
                  </p>
                </div>
              </div>
            )}

            {selectedCandidate && !selectedIsCurrent && (
              <div className="split-comparison-panel">
                <div className="split-comparison-header">
                  <div>
                    <span>ALTERNATIVE SPLIT COMPARISON</span>

                    <h4>Current vs Candidate</h4>
                  </div>

                  <div
                    className={`split-comparison-delta ${
                      giniChange > 0
                        ? "improved"
                        : giniChange < 0
                          ? "worse"
                          : "same"
                    }`}
                  >
                    {giniChange > 0
                      ? "Lower Gini"
                      : giniChange < 0
                        ? "Higher Gini"
                        : "Same Gini"}
                  </div>
                </div>

                <div className="split-comparison-grid">
                  <div className="split-comparison-card">
                    <span>CURRENT TREE SPLIT</span>

                    <strong>
                      {activeNode.feature} ≤{" "}
                      {Number(currentCandidate.threshold).toFixed(2)}
                    </strong>

                    <div className="split-comparison-stat">
                      <span>Weighted Gini</span>

                      <b>{Number(currentCandidate.weighted_gini).toFixed(4)}</b>
                    </div>

                    <div className="split-comparison-stat">
                      <span>Left / Right</span>

                      <b>
                        {currentCandidate.left_count}
                        {" / "}
                        {currentCandidate.right_count}
                      </b>
                    </div>
                  </div>

                  <div className="split-comparison-arrow">→</div>

                  <div className="split-comparison-card alternative">
                    <span>SELECTED ALTERNATIVE</span>

                    <strong>
                      {activeNode.feature} ≤{" "}
                      {Number(selectedCandidate.threshold).toFixed(2)}
                    </strong>

                    <div className="split-comparison-stat">
                      <span>Weighted Gini</span>

                      <b>
                        {Number(selectedCandidate.weighted_gini).toFixed(4)}
                      </b>
                    </div>

                    <div className="split-comparison-stat">
                      <span>Left / Right</span>

                      <b>
                        {selectedCandidate.left_count}
                        {" / "}
                        {selectedCandidate.right_count}
                      </b>
                    </div>
                  </div>
                </div>

                <div className="split-comparison-changes">
                  <div>
                    <span>Gini change</span>

                    <strong>
                      {giniChange >= 0 ? "+" : ""}
                      {giniChange.toFixed(4)}
                    </strong>
                  </div>

                  <div>
                    <span>Left rows</span>

                    <strong>
                      {leftChange >= 0 ? "+" : ""}
                      {leftChange}
                    </strong>
                  </div>

                  <div>
                    <span>Right rows</span>

                    <strong>
                      {rightChange >= 0 ? "+" : ""}
                      {rightChange}
                    </strong>
                  </div>
                </div>

                <div className="split-comparison-explanation">
                  {giniChange > 0 ? (
                    <>
                      <strong>
                        This candidate produces a lower weighted Gini.
                      </strong>

                      <p>
                        Compared with the split currently used by the tree, this
                        threshold creates a purer child-group arrangement on the
                        training rows reaching this node.
                      </p>
                    </>
                  ) : giniChange < 0 ? (
                    <>
                      <strong>
                        This candidate produces a higher weighted Gini.
                      </strong>

                      <p>
                        The alternative threshold creates a less pure
                        child-group arrangement than the split currently used by
                        the tree.
                      </p>
                    </>
                  ) : (
                    <>
                      <strong>
                        Both splits produce the same weighted Gini.
                      </strong>

                      <p>
                        The threshold changes the row allocation, but the
                        resulting weighted impurity is unchanged.
                      </p>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="split-explorer-footer">
          Showing {candidates.length} candidate thresholds for Node{" "}
          {activeNode.node_id}.
        </div>
      </div>
    </section>
  );
}

function App() {
  const [activeWorkspace, setActiveWorkspace] = useState("tree");
  const [activeView, setActiveView] = useState("tree");

  const [file, setFile] = useState(null);

  const [target, setTarget] = useState("target");

  const [maxDepth, setMaxDepth] = useState(4);

  const [minSamplesLeaf, setMinSamplesLeaf] = useState(1);

  const [ccpAlpha, setCcpAlpha] = useState(0);

  const [sampleHeatMode, setSampleHeatMode] = useState(false);

  const [errorFocusMode, setErrorFocusMode] = useState(false);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  const [result, setResult] = useState(null);

  const [selectedNode, setSelectedNode] = useState(null);

  const [selectedRow, setSelectedRow] = useState(null);

  const [selectedConfusionCell, setSelectedConfusionCell] = useState(null);

  const [linkedFeature, setLinkedFeature] = useState(null);

  const [predictionPath, setPredictionPath] = useState(null);

  const [predictionLoading, setPredictionLoading] = useState(false);

  useEffect(() => {
    if (!predictionPath) {
      return;
    }

    const timer = setTimeout(() => {
      const section = document.getElementById("prediction-path-section");

      if (section) {
        section.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    }, 100);

    return () => clearTimeout(timer);
  }, [predictionPath]);

  async function trainModel() {
    setError("");

    if (!file) {
      setError("Please upload a CSV file.");

      return;
    }

    if (!target.trim()) {
      setError("Please enter the target column.");

      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();

      formData.append("file", file);

      const url = `http://127.0.0.1:8000/train?target=${encodeURIComponent(
        target,
      )}&max_depth=${maxDepth}&min_samples_leaf=${minSamplesLeaf}&ccp_alpha=${ccpAlpha}`;

      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Training failed.");
      }

      setResult(data);
      setActiveView("tree");

      const root = data.tree.find((node) => node.id === 0);

      setSelectedNode(root || null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function predictRow(row) {
    if (!row || !result) {
      return;
    }

    setPredictionLoading(true);
    setError("");

    try {
      const features = {};

      result.dataset.features.forEach((feature) => {
        features[feature] = row[feature];
      });

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          features,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setPredictionPath(data);
      setActiveView("data");
    } catch (err) {
      setError(err.message);
    } finally {
      setPredictionLoading(false);
    }
  }

  function handleFileChange(event) {
    const selected = event.target.files?.[0];

    if (selected) {
      setFile(selected);

      setResult(null);

      setSelectedNode(null);

      setError("");
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon">M</div>

          <div>
            <h1>ModelMind</h1>

            <span>ML Visualization Lab</span>
          </div>
        </div>

        <nav className="workspace-nav" aria-label="Workspace">
          <button
            type="button"
            className={activeWorkspace === "tree" ? "active" : ""}
            aria-current={activeWorkspace === "tree" ? "page" : undefined}
            onClick={() => setActiveWorkspace("tree")}
          >
            Decision Tree
          </button>
          <button
            type="button"
            className={activeWorkspace === "lab" ? "active" : ""}
            aria-current={activeWorkspace === "lab" ? "page" : undefined}
            onClick={() => setActiveWorkspace("lab")}
          >
            Model Lab
          </button>
        </nav>
      </header>

      {activeWorkspace === "lab" && (
        <main className="lab-main">
          <P3Lab file={file} target={target} />
        </main>
      )}

      {activeWorkspace === "tree" && (
        <main>
          <section className="hero">
            <div className="eyebrow">MACHINE LEARNING • VISUAL EXPLORER</div>

            <h2>
              See how your <span>Decision Tree</span> thinks.
            </h2>

            <p>
              Upload a dataset and explore how the model learns, splits data,
              and makes predictions.
            </p>
          </section>

          <section className="training-panel">
            <div className="upload-area">
              <input
                id="csv-upload"
                type="file"
                accept=".csv"
                hidden
                onChange={handleFileChange}
              />

              <label htmlFor="csv-upload" className="upload-label">
                <div className="upload-icon">↑</div>

                <div>
                  <strong>{file ? file.name : "Upload CSV dataset"}</strong>

                  <span>Click to choose a .csv file</span>
                </div>
              </label>
            </div>

            <div className="field">
              <label>Target column</label>

              <input
                value={target}
                onChange={(event) => setTarget(event.target.value)}
                placeholder="target"
              />
            </div>

            <div className="field">
              <label>Max depth</label>

              <input
                type="number"
                min="1"
                max="20"
                value={maxDepth}
                onChange={(event) => setMaxDepth(Number(event.target.value))}
              />
            </div>
            <div className="field">
              <label>Minimum samples per leaf</label>

              <input
                type="number"
                min="1"
                max="10"
                value={minSamplesLeaf}
                onChange={(event) =>
                  setMinSamplesLeaf(Math.max(1, Number(event.target.value)))
                }
              />
            </div>

            <div className="field">
              <label>Pruning alpha</label>

              <input
                type="number"
                min="0"
                max="0.1"
                step="0.001"
                value={ccpAlpha}
                onChange={(event) =>
                  setCcpAlpha(Math.max(0, Number(event.target.value)))
                }
              />
            </div>

            <div className="p2-toggle-row">
              <label>
                <input
                  type="checkbox"
                  checked={sampleHeatMode}
                  onChange={(event) => setSampleHeatMode(event.target.checked)}
                />
                Node sample heat
              </label>

              <label>
                <input
                  type="checkbox"
                  checked={errorFocusMode}
                  onChange={(event) => setErrorFocusMode(event.target.checked)}
                />
                Error focus
              </label>
            </div>

            <button
              className="train-button"
              onClick={trainModel}
              disabled={loading}
            >
              {loading ? "Training..." : "Train Tree"}
            </button>
          </section>

          {error && <div className="error-box">{error}</div>}

          {result && (
            <>
              <nav
                className="result-tabs"
                aria-label="Decision tree views"
                role="tablist"
              >
                {[
                  ["tree", "Tree"],
                  ["data", "Data"],
                  ["analysis", "Analysis"],
                ].map(([view, label]) => (
                  <button
                    key={view}
                    type="button"
                    role="tab"
                    aria-selected={activeView === view}
                    className={activeView === view ? "active" : ""}
                    onClick={() => setActiveView(view)}
                  >
                    {label}
                  </button>
                ))}
                <span className="result-tabs-summary">
                  {result.dataset.rows} rows · {result.dataset.features.length}{" "}
                  features · {result.class_names.length} classes
                </span>
              </nav>

              {activeView === "data" && (
                <section className="section">
                  <div className="section-title">
                    <span>DATASET</span>

                    <h3>Dataset Overview</h3>
                  </div>

                  <div className="stats-grid">
                    <StatCard label="Rows" value={result.dataset.rows} />

                    <StatCard label="Columns" value={result.dataset.columns} />

                    <StatCard
                      label="Training rows"
                      value={result.dataset.train_rows}
                    />

                    <StatCard
                      label="Testing rows"
                      value={result.dataset.test_rows}
                    />
                  </div>

                  <div className="features">
                    <strong>Features</strong>

                    <div>
                      {result.dataset.features.map((feature) => (
                        <span key={feature}>{feature}</span>
                      ))}
                    </div>
                  </div>
                </section>
              )}

              {activeView === "analysis" && (
                <section className="section">
                  <div className="section-title">
                    <span>EVALUATION</span>

                    <h3>Model Performance</h3>
                  </div>

                  <div className="metrics-grid">
                    <MetricCard
                      label="Accuracy"
                      value={result.metrics.accuracy}
                    />

                    <MetricCard
                      label="Precision"
                      value={result.metrics.precision}
                    />

                    <MetricCard label="Recall" value={result.metrics.recall} />

                    <MetricCard label="F1 Score" value={result.metrics.f1} />
                  </div>
                </section>
              )}

              {activeView === "data" && (
                <section className="section row-explorer-section">
                  <div className="section-title">
                    <span>DATA EXPLORER</span>

                    <h3>Prediction Row Explorer</h3>

                    <p>
                      Select any dataset row to replay its decision path through
                      the tree.
                    </p>
                  </div>

                  <div className="row-explorer">
                    <div className="row-explorer-header">
                      <strong>Dataset Rows</strong>

                      <span>{result.dataset.rows_data?.length || 0} rows</span>
                    </div>

                    <div className="row-table-wrapper">
                      <table className="row-table">
                        <thead>
                          <tr>
                            <th>#</th>

                            {result.dataset.features.map((feature) => (
                              <th key={feature}>{feature}</th>
                            ))}

                            <th>Target</th>

                            <th>Action</th>
                          </tr>
                        </thead>

                        <tbody>
                          {(result.dataset.rows_data || []).map(
                            (row, index) => (
                              <tr
                                key={index}
                                className={
                                  selectedRow === index ? "selected-row" : ""
                                }
                              >
                                <td>{index + 1}</td>

                                {result.dataset.features.map((feature) => (
                                  <td key={feature}>
                                    {typeof row[feature] === "number"
                                      ? Number(row[feature]).toFixed(2)
                                      : row[feature]}
                                  </td>
                                ))}

                                <td>
                                  <span className="target-badge">
                                    {row[result.dataset.target]}
                                  </span>
                                </td>

                                <td>
                                  <button
                                    className="row-predict-button"
                                    onClick={() => {
                                      setSelectedRow(index);
                                      predictRow(row);
                                    }}
                                    disabled={predictionLoading}
                                  >
                                    {selectedRow === index && predictionLoading
                                      ? "Predicting..."
                                      : "Trace Path"}
                                  </button>
                                </td>
                              </tr>
                            ),
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </section>
              )}

              {activeView === "data" && predictionPath && (
                <section
                  id="prediction-path-section"
                  className="section prediction-path-section"
                >
                  <div className="section-title">
                    <span>EXPLAINABILITY</span>

                    <h3>Prediction Path</h3>

                    <p>
                      Follow the exact rules this row passed through to reach
                      its prediction.
                    </p>
                  </div>

                  <div className="prediction-result-card">
                    <div className="prediction-summary">
                      <div>
                        <span>FINAL PREDICTION</span>

                        <strong>{predictionPath.prediction_name}</strong>
                      </div>

                      <div>
                        <span>PATH LENGTH</span>

                        <strong>{predictionPath.path.length} steps</strong>
                      </div>
                    </div>

                    <div className="prediction-timeline">
                      {predictionPath.path.map((step, index) => {
                        if (step.type === "leaf") {
                          return (
                            <div
                              className="prediction-step prediction-leaf"
                              key={step.node_id}
                            >
                              <div className="prediction-step-number">✓</div>

                              <div className="prediction-step-content">
                                <span>FINAL LEAF · NODE {step.node_id}</span>

                                <strong>{step.prediction_name}</strong>

                                <small>
                                  {step.samples} training samples reached this
                                  leaf.
                                </small>
                              </div>
                            </div>
                          );
                        }

                        return (
                          <div className="prediction-step" key={step.node_id}>
                            <div className="prediction-step-number">
                              {index + 1}
                            </div>

                            <div className="prediction-step-content">
                              <span>NODE {step.node_id}</span>

                              <strong>
                                {step.feature} ≤{" "}
                                {Number(step.threshold).toFixed(2)}
                              </strong>

                              <div className="prediction-comparison">
                                <span>
                                  Row value:
                                  <b> {Number(step.value).toFixed(2)}</b>
                                </span>

                                <span
                                  className={
                                    step.result
                                      ? "decision-true"
                                      : "decision-false"
                                  }
                                >
                                  {step.result ? "✓ TRUE" : "✕ FALSE"}
                                </span>
                              </div>

                              <small>
                                → Follow the <b>{step.branch}</b> branch to node{" "}
                                {step.next_node}.
                              </small>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </section>
              )}

              {activeView === "tree" && (
                <section className="section tree-section">
                  <div className="section-title">
                    <span>MODEL</span>

                    <h3>Decision Tree</h3>

                    <p>Click any node to inspect it.</p>
                  </div>

                  <div className="tree-container">
                    <DecisionTree
                      nodes={result.tree}
                      selectedNode={selectedNode}
                      onSelect={setSelectedNode}
                      predictionPath={predictionPath}
                      linkedFeature={linkedFeature}
                    />
                  </div>
                </section>
              )}

              {activeView === "tree" && selectedNode && (
                <section className="section">
                  <div className="section-title">
                    <span>EXPLAINABILITY</span>

                    <h3>Node Inspector</h3>
                  </div>

                  <div className="inspector">
                    <div>
                      <div className="node-label">
                        {selectedNode.is_leaf ? "LEAF NODE" : "DECISION NODE"}
                      </div>

                      <div className="equation">
                        {selectedNode.is_leaf
                          ? `Prediction: ${selectedNode.prediction_name}`
                          : `${selectedNode.feature} ≤ ${selectedNode.threshold?.toFixed(4)}`}
                      </div>
                    </div>

                    <div className="details">
                      <div>
                        <span>Samples</span>
                        <strong>{selectedNode.samples}</strong>
                      </div>

                      <div>
                        <span>Gini</span>
                        <strong>{selectedNode.gini}</strong>
                      </div>

                      <div>
                        <span>Prediction</span>
                        <strong>{selectedNode.prediction_name}</strong>
                      </div>
                    </div>
                  </div>

                  <h4>Class Distribution</h4>

                  {selectedNode.class_distribution.map((value, index) => {
                    const percentage = selectedNode.samples
                      ? (value / selectedNode.samples) * 100
                      : 0;

                    return (
                      <div className="distribution" key={index}>
                        <span>{result.class_names[index]}</span>

                        <div className="distribution-track">
                          <div
                            className="distribution-fill"
                            style={{
                              width: `${percentage}%`,
                            }}
                          />
                        </div>

                        <strong>{value}</strong>
                      </div>
                    );
                  })}
                </section>
              )}

              {activeView === "analysis" && (
                <section className="section">
                  <div className="section-title">
                    <span>EXPLANATION</span>

                    <h3>Feature Importance</h3>
                  </div>

                  <div className="importance">
                    {result.feature_importance.map((item) => (
                      <div
                        className={`importance-row ${
                          linkedFeature === item.feature
                            ? "importance-row-linked"
                            : ""
                        }`}
                        key={item.feature}
                        onClick={() =>
                          setLinkedFeature(
                            linkedFeature === item.feature
                              ? null
                              : item.feature,
                          )
                        }
                      >
                        <span>{item.feature}</span>

                        <div className="importance-track">
                          <div
                            className="importance-fill"
                            style={{
                              width: `${item.importance * 100}%`,
                            }}
                          />
                        </div>

                        <strong>{(item.importance * 100).toFixed(1)}%</strong>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {activeView === "analysis" && (
                <section className="section">
                  <div className="section-title">
                    <span>CLASSIFICATION</span>

                    <h3>Confusion Matrix</h3>
                  </div>

                  <div className="matrix-wrapper">
                    <table>
                      <thead>
                        <tr>
                          <th>Actual / Predicted</th>

                          {result.class_names.map((name) => (
                            <th key={name}>{name}</th>
                          ))}
                        </tr>
                      </thead>

                      <tbody>
                        {result.confusion_matrix.map((row, rowIndex) => (
                          <tr key={rowIndex}>
                            <th>{result.class_names[rowIndex]}</th>

                            {row.map((value, columnIndex) => (
                              <td
                                key={columnIndex}
                                className={`confusion-cell ${
                                  rowIndex === columnIndex ? "correct" : ""
                                } ${
                                  selectedConfusionCell &&
                                  selectedConfusionCell.actual === rowIndex &&
                                  selectedConfusionCell.predicted ===
                                    columnIndex
                                    ? "confusion-cell-selected"
                                    : ""
                                }`}
                                onClick={() => {
                                  setSelectedConfusionCell({
                                    actual: rowIndex,
                                    predicted: columnIndex,
                                  });
                                }}
                              >
                                {value}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {selectedConfusionCell && (
                    <div className="confusion-rows-panel">
                      <div className="confusion-rows-header">
                        <div>
                          <span className="confusion-rows-label">
                            SELECTED CONFUSION CELL
                          </span>

                          <h4>
                            {result.class_names[selectedConfusionCell.actual]} →{" "}
                            {
                              result.class_names[
                                selectedConfusionCell.predicted
                              ]
                            }
                          </h4>
                        </div>

                        <strong>
                          {
                            (result.confusion_rows || []).filter(
                              (item) =>
                                item.actual === selectedConfusionCell.actual &&
                                item.predicted ===
                                  selectedConfusionCell.predicted,
                            ).length
                          }{" "}
                          rows
                        </strong>
                      </div>

                      {(() => {
                        const matchingRows = (
                          result.confusion_rows || []
                        ).filter(
                          (item) =>
                            item.actual === selectedConfusionCell.actual &&
                            item.predicted === selectedConfusionCell.predicted,
                        );

                        if (matchingRows.length === 0) {
                          return (
                            <div className="confusion-empty">
                              No test rows belong to this confusion-matrix cell.
                            </div>
                          );
                        }

                        return (
                          <div className="confusion-row-list">
                            {matchingRows.map((item) => (
                              <div
                                className="confusion-row-item"
                                key={item.row_index}
                              >
                                <strong>
                                  Dataset row {item.row_index + 1}
                                </strong>

                                <span>Actual: {item.actual_name}</span>

                                <span>Predicted: {item.predicted_name}</span>

                                <button
                                  className="row-predict-button"
                                  onClick={() => {
                                    const row =
                                      result.dataset.rows_data[item.row_index];

                                    if (row) {
                                      setSelectedRow(item.row_index);

                                      predictRow(row);
                                    }
                                  }}
                                >
                                  Trace Path
                                </button>
                              </div>
                            ))}
                          </div>
                        );
                      })()}
                    </div>
                  )}
                </section>
              )}
            </>
          )}

          {!result && !loading && (
            <div className="empty-state">
              <div className="empty-icon">◇</div>

              <h3>Your ML visualization starts here</h3>

              <p>
                Upload a classification CSV, select the target column, and train
                your first Decision Tree.
              </p>
            </div>
          )}

          {result?.split_explorer &&
            activeView === "analysis" &&
            Object.keys(result.split_explorer).length > 0 && (
              <SplitExplorer
                splitExplorer={result.split_explorer}
                selectedNode={selectedNode}
                onSelectNode={setSelectedNode}
              />
            )}
        </main>
      )}

      <footer>
        ModelMind · Interactive Machine Learning Visualization Lab
      </footer>
    </div>
  );
}

export default App;
