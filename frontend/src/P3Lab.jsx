import React, { useMemo, useState } from "react";
import { API_BASE_URL } from "./api.js";

const MODELS = [
  { value: "decision_tree", label: "Decision Tree" },
  { value: "logistic", label: "Logistic Regression" },
  { value: "knn", label: "KNN" },
  { value: "svm", label: "SVM" },
  { value: "random_forest", label: "Random Forest" }
];

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${(number * 100).toFixed(1)}%`;
}

function formatNumber(value, digits = 3) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number.toFixed(digits);
}

function normalizeEffects(effects, features) {
  if (!effects) return [];

  if (Array.isArray(effects)) {
    return effects
      .map((item, index) => {
        if (typeof item === "number") {
          return {
            feature: features?.[index] || `Feature ${index + 1}`,
            value: item
          };
        }

        return {
          feature:
            item.feature ||
            item.name ||
            features?.[index] ||
            `Feature ${index + 1}`,
          value:
            item.value ??
            item.importance ??
            item.coefficient ??
            0
        };
      })
      .sort((a, b) => Math.abs(Number(b.value)) - Math.abs(Number(a.value)));
  }

  return Object.entries(effects)
    .map(([feature, value]) => ({
      feature,
      value: Number(value) || 0
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
}

function normalizeProbabilities(probabilities) {
  if (!Array.isArray(probabilities)) return [];

  return probabilities.map((row, index) => {
    if (row && typeof row === "object") {
      return {
        row: row.row ?? row.row_index ?? row.index ?? index + 1,
        predicted:
          row.predicted ??
          row.prediction ??
          row.class ??
          "—",
        probability:
          row.probability ??
          row.confidence ??
          row.max_probability ??
          null,
        probabilities: row.probabilities || row.scores || null
      };
    }

    return {
      row: index + 1,
      predicted: "—",
      probability: null,
      probabilities: null
    };
  });
}

export default function P3Lab({ file, target }) {
  const [model, setModel] = useState("logistic");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedProbabilityRow, setSelectedProbabilityRow] = useState(0);

  const effects = useMemo(
    () => normalizeEffects(result?.feature_effects, result?.features),
    [result]
  );

  const probabilities = useMemo(
    () => normalizeProbabilities(result?.probabilities),
    [result]
  );

  const maxEffect = useMemo(() => {
    if (!effects.length) return 1;

    return Math.max(
      ...effects.map(item => Math.abs(Number(item.value) || 0)),
      1
    );
  }, [effects]);

  async function analyzeModel() {
    if (!file) {
      setError("Please upload a CSV dataset first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const url =
        `${API_BASE_URL}/p3/analyze` +
        `?target=${encodeURIComponent(target)}` +
        `&model_name=${encodeURIComponent(model)}`;

      const response = await fetch(url, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
          data?.message ||
          "P3 analysis failed."
        );
      }

      setResult(data);
      setSelectedProbabilityRow(0);
    } catch (err) {
      setError(
        err?.message ||
        "Unable to connect to the ML backend."
      );
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  const selectedProbability =
    probabilities[selectedProbabilityRow] || null;

  const projectionPoints =
    result?.projection?.points || [];

  const boundaryPoints =
    result?.projection?.boundary || [];

  const metrics = result?.metrics || {};

  return (
    <section className="section p3-lab">
      <div className="p3-header">
        <div>
          <div className="p3-kicker">
            MODEL UNDERSTANDING
          </div>

          <h2>
            Model Understanding Lab
          </h2>

          <p>
            Compare different machine learning models
            and visualize how each model learns from
            your dataset.
          </p>
        </div>
      </div>

      <div className="p3-controls">
        <div className="field">
          <label>
            Model
          </label>

          <select
            value={model}
            onChange={event =>
              setModel(event.target.value)
            }
            disabled={loading}
          >
            {MODELS.map(item => (
              <option
                key={item.value}
                value={item.value}
              >
                {item.label}
              </option>
            ))}
          </select>
        </div>

        <button
          className="train-button"
          onClick={analyzeModel}
          disabled={loading || !file}
        >
          {loading
            ? "Analyzing..."
            : "Run Model Analysis"}
        </button>
      </div>

      {!file && (
        <p className="p3-muted">
          Upload a CSV file and train the Decision Tree
          first, then run a model analysis here.
        </p>
      )}

      {error && (
        <div className="error-panel">
          {error}
        </div>
      )}

      {result && (
        <div className="p3-results">
          <div className="p3-model-summary">
            <strong>
              {MODELS.find(item => item.value === result.model)?.label ||
                result.model}
            </strong>

            <span>
              {result.dataset?.rows ?? "—"} rows
            </span>

            <span>
              {result.features?.length ?? "—"} features
            </span>

            <span>
              {result.classes?.length ?? "—"} classes
            </span>
          </div>

          <div className="p3-metrics">
            {[
              ["Accuracy", metrics.accuracy],
              ["Precision", metrics.precision],
              ["Recall", metrics.recall],
              ["F1 Score", metrics.f1]
            ].map(([label, value]) => (
              <div
                className="p3-metric-card"
                key={label}
              >
                <span>{label}</span>
                <strong>
                  {formatPercent(value)}
                </strong>
              </div>
            ))}
          </div>

          <div className="p3-panel">
            <h3>
              Feature Effects
            </h3>

            <p className="p3-muted">
              How strongly each feature contributes to
              this model.
            </p>

            <div className="p3-feature-list">
              {effects.length ? (
                effects.map(item => (
                  <div
                    className="p3-feature-row"
                    key={item.feature}
                  >
                    <strong>
                      {item.feature}
                    </strong>

                    <div className="p3-bar">
                      <div
                        style={{
                          width: `${Math.min(
                            100,
                            Math.abs(Number(item.value) || 0) /
                              maxEffect *
                              100
                          )}%`
                        }}
                      />
                    </div>

                    <span>
                      {formatNumber(item.value, 4)}
                    </span>
                  </div>
                ))
              ) : (
                <p className="p3-muted">
                  No feature-effect data returned.
                </p>
              )}
            </div>
          </div>

          <div className="p3-panel">
            <h3>
              Prediction Probability
            </h3>

            <p className="p3-muted">
              Inspect the predictions and confidence
              returned by the selected model.
            </p>

            {probabilities.length ? (
              <>
                <div className="p3-probability-list">
                  {probabilities.slice(0, 12).map((item, index) => (
                    <button
                      type="button"
                      className="p3-probability-row"
                      key={`${item.row}-${index}`}
                      onClick={() =>
                        setSelectedProbabilityRow(index)
                      }
                    >
                      <strong>
                        Row {item.row}
                      </strong>

                      <span>
                        {item.predicted}
                      </span>

                      <div className="p3-probability-track">
                        <div
                          style={{
                            width:
                              item.probability == null
                                ? "0%"
                                : `${Math.min(
                                    100,
                                    Number(item.probability) * 100
                                  )}%`
                          }}
                        />
                      </div>

                      <span>
                        {item.probability == null
                          ? "—"
                          : formatPercent(item.probability)}
                      </span>
                    </button>
                  ))}
                </div>

                {selectedProbability && (
                  <div className="p3-prediction">
                    <strong>
                      Row {selectedProbability.row}
                    </strong>

                    <div className="p3-prediction-meta">
                      <span>
                        Prediction
                      </span>

                      <strong>
                        {selectedProbability.predicted}
                      </strong>

                      <span>
                        Confidence
                      </span>

                      <strong>
                        {selectedProbability.probability == null
                          ? "—"
                          : formatPercent(
                              selectedProbability.probability
                            )}
                      </strong>
                    </div>

                    {selectedProbability.probabilities && (
                      <div className="p3-info-grid">
                        {Object.entries(
                          selectedProbability.probabilities
                        ).map(([label, value]) => (
                          <div key={label}>
                            <span>{label}</span>
                            <strong>
                              {formatPercent(value)}
                            </strong>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <p className="p3-muted">
                Probability information is not available
                for this model.
              </p>
            )}
          </div>

          <div className="p3-panel">
            <h3>
              2D Model Visualization
            </h3>

            <p className="p3-muted">
              The dataset is projected into two dimensions
              so you can visually inspect class separation.
            </p>

            <div className="p3-projection">
              {projectionPoints.length ? (
                projectionPoints.map((point, index) => {
                  const x = Number(point.x ?? point.pca1 ?? 0);
                  const y = Number(point.y ?? point.pca2 ?? 0);

                  const minX = Math.min(
                    ...projectionPoints.map(
                      item => Number(item.x ?? item.pca1 ?? 0)
                    )
                  );

                  const maxX = Math.max(
                    ...projectionPoints.map(
                      item => Number(item.x ?? item.pca1 ?? 0)
                    )
                  );

                  const minY = Math.min(
                    ...projectionPoints.map(
                      item => Number(item.y ?? item.pca2 ?? 0)
                    )
                  );

                  const maxY = Math.max(
                    ...projectionPoints.map(
                      item => Number(item.y ?? item.pca2 ?? 0)
                    )
                  );

                  const px =
                    10 +
                    ((x - minX) / ((maxX - minX) || 1)) * 80;

                  const py =
                    90 -
                    ((y - minY) / ((maxY - minY) || 1)) * 80;

                  return (
                    <span
                      className="p3-point"
                      key={index}
                      title={`${point.class ?? point.label ?? ""}`}
                      style={{
                        left: `${px}%`,
                        top: `${py}%`
                      }}
                    />
                  );
                })
              ) : (
                <div className="p3-projection-info">
                  Projection data is not available.
                </div>
              )}

              {boundaryPoints.length > 0 && (
                <div className="p3-projection-info">
                  Decision-region boundary available
                </div>
              )}
            </div>

            {result?.projection?.explained_variance && (
              <div className="p3-projection-info">
                Explained variance:{" "}
                {result.projection.explained_variance
                  .map(value =>
                    `${(Number(value) * 100).toFixed(1)}%`
                  )
                  .join(" + ")}
              </div>
            )}
          </div>

          {result.knn_neighbors?.length > 0 && (
            <div className="p3-panel">
              <h3>
                KNN Nearest Neighbors
              </h3>

              <div className="p3-neighbors">
                {result.knn_neighbors.map((neighbor, index) => (
                  <div
                    className="p3-neighbor"
                    key={index}
                  >
                    <strong>
                      Neighbor {index + 1}
                    </strong>

                    <span>
                      Row:{" "}
                      {neighbor.row ??
                        neighbor.index ??
                        "—"}
                    </span>

                    <span>
                      Distance:{" "}
                      {formatNumber(
                        neighbor.distance,
                        4
                      )}
                    </span>

                    <span>
                      Class:{" "}
                      {neighbor.class ??
                        neighbor.label ??
                        "—"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.svm && (
            <div className="p3-panel">
              <h3>
                SVM Support Vector Information
              </h3>

              <div className="p3-info-grid">
                <div>
                  <span>
                    Support vectors
                  </span>

                  <strong>
                    {result.svm.support_vector_count ??
                      result.svm.count ??
                      "—"}
                  </strong>
                </div>

                <div>
                  <span>
                    Kernel
                  </span>

                  <strong>
                    {result.svm.kernel ?? "RBF"}
                  </strong>
                </div>

                <div>
                  <span>
                    Classes
                  </span>

                  <strong>
                    {result.svm.classes?.length ??
                      result.classes?.length ??
                      "—"}
                  </strong>
                </div>
              </div>
            </div>
          )}

          {result.random_forest && (
            <div className="p3-panel">
              <h3>
                Random Forest Voting
              </h3>

              <div className="p3-info-grid">
                <div>
                  <span>
                    Trees
                  </span>

                  <strong>
                    {result.random_forest.tree_count ??
                      result.random_forest.n_estimators ??
                      "—"}
                  </strong>
                </div>

                <div>
                  <span>
                    Average tree depth
                  </span>

                  <strong>
                    {formatNumber(
                      result.random_forest.average_depth,
                      2
                    )}
                  </strong>
                </div>

                <div>
                  <span>
                    Voting information
                  </span>

                  <strong>
                    Available
                  </strong>
                </div>
              </div>

              {result.random_forest.voting?.length > 0 && (
                <div className="p3-voting">
                  {result.random_forest.voting
                    .slice(0, 10)
                    .map((vote, index) => (
                      <div
                        className="p3-vote-row"
                        key={index}
                      >
                        <strong>
                          Row {vote.row ?? index + 1}
                        </strong>

                        <span>
                          {vote.prediction ??
                            vote.predicted ??
                            "—"}
                        </span>

                        <span>
                          {vote.votes
                            ? Object.entries(vote.votes)
                                .map(
                                  ([name, count]) =>
                                    `${name}: ${count}`
                                )
                                .join(" · ")
                            : vote.vote_counts
                              ? String(vote.vote_counts)
                              : "—"}
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}

          {result.confusion_matrix && (
            <div className="p3-panel">
              <h3>
                Model Confusion Matrix
              </h3>

              <div className="p3-info-grid">
                {result.confusion_matrix.flatMap(
                  (row, rowIndex) =>
                    row.map((value, colIndex) => (
                      <div
                        key={`${rowIndex}-${colIndex}`}
                      >
                        <span>
                          {result.classes?.[rowIndex] ??
                            `Actual ${rowIndex}`}{" "}
                          →{" "}
                          {result.classes?.[colIndex] ??
                            `Predicted ${colIndex}`}
                        </span>

                        <strong>
                          {value}
                        </strong>
                      </div>
                    ))
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
