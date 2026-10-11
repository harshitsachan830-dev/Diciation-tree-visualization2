from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np

from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


app = FastAPI(
    title="ModelMind ML Visualization API"
)


# Stores the latest trained model and preprocessing state.
# This is intentionally simple for the local ModelMind app.
MODEL_STATE = {
    "model": None,
    "feature_names": [],
    "feature_encoders": {},
    "target_encoder": None,
    "class_names": [],
}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "ModelMind ML Visualization API is running",
        "status": "ok",
    }


class PredictionRequest(BaseModel):
    features: dict


def trace_tree_prediction(model, feature_names, sample):
    tree = model.tree_

    values = np.array([
        float(sample[feature])
        for feature in feature_names
    ])

    node_id = 0
    path = []

    while True:
        feature_index = tree.feature[node_id]
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])

        if feature_index == -2:
            predicted_class = int(
                np.argmax(tree.value[node_id][0])
            )

            path.append({
                "node_id": int(node_id),
                "type": "leaf",
                "prediction": predicted_class,
                "samples": int(tree.n_node_samples[node_id]),
            })

            return path

        feature_name = feature_names[feature_index]
        feature_value = float(
            values[feature_names.index(feature_name)]
        )

        threshold = float(tree.threshold[node_id])
        goes_left = feature_value <= threshold
        next_node = left if goes_left else right

        path.append({
            "node_id": int(node_id),
            "type": "split",
            "feature": feature_name,
            "threshold": threshold,
            "value": feature_value,
            "operator": "<=",
            "result": bool(goes_left),
            "branch": "left" if goes_left else "right",
            "next_node": int(next_node),
        })

        node_id = next_node


@app.post("/predict")
async def predict_sample(request: PredictionRequest):

    model = MODEL_STATE["model"]
    feature_names = MODEL_STATE["feature_names"]

    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Train a model first.",
        )

    sample = request.features

    missing = [
        feature
        for feature in feature_names
        if feature not in sample
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}",
        )

    encoded_values = []

    for feature in feature_names:

        value = sample[feature]

        encoder = MODEL_STATE["feature_encoders"].get(feature)

        if encoder is not None:

            try:
                value = encoder.transform([str(value)])[0]

            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unknown value '{value}' "
                        f"for feature '{feature}'. "
                        f"Allowed values: "
                        f"{list(encoder.classes_)}"
                    ),
                )

        else:

            try:
                value = float(value)

            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Feature '{feature}' must be numeric."
                    ),
                )

        encoded_values.append(float(value))

    sample_array = np.array(
        [encoded_values],
        dtype=float,
    )

    prediction = int(
        model.predict(sample_array)[0]
    )

    path = trace_tree_prediction(
        model,
        feature_names,
        dict(
            zip(
                feature_names,
                encoded_values,
            )
        ),
    )

    class_names = MODEL_STATE["class_names"]

    for step in path:

        if step["type"] == "leaf":

            step["prediction_name"] = (
                class_names[step["prediction"]]
            )

    return {
        "prediction": prediction,
        "prediction_name": class_names[prediction],
        "path": path,
    }


def build_split_explorer(model, X_train, y_train, feature_names, class_names):
    """Build candidate split information for Decision Tree explainability."""

    tree = model.tree_
    explorer = {}

    decision_path = model.decision_path(X_train)

    for node_id in range(tree.node_count):
        feature_index = int(tree.feature[node_id])

        if feature_index < 0:
            continue

        feature = feature_names[feature_index]

        row_indices = decision_path[:, node_id].nonzero()[0]

        if len(row_indices) == 0:
            continue

        values = X_train.iloc[row_indices][feature].astype(float)
        targets = y_train[row_indices]

        unique_values = sorted(values.unique().tolist())

        if len(unique_values) < 2:
            continue

        candidates = []

        for left_value, right_value in zip(
            unique_values[:-1],
            unique_values[1:],
        ):
            threshold = (left_value + right_value) / 2.0

            left_mask = values <= threshold
            right_mask = values > threshold

            left_targets = targets[left_mask]
            right_targets = targets[right_mask]

            def distribution(series):
                counts = [
                    int((series == class_id).sum())
                    for class_id in range(len(class_names))
                ]
                return counts

            def gini(series):
                if len(series) == 0:
                    return 0.0

                probabilities = [
                    float((series == class_id).sum()) / len(series)
                    for class_id in range(len(class_names))
                ]

                return 1.0 - sum(
                    probability ** 2
                    for probability in probabilities
                )

            total = len(targets)

            weighted_gini = (
                len(left_targets) * gini(left_targets)
                + len(right_targets) * gini(right_targets)
            ) / total

            candidates.append(
                {
                    "threshold": round(float(threshold), 6),
                    "left_count": int(len(left_targets)),
                    "right_count": int(len(right_targets)),
                    "left_distribution": distribution(left_targets),
                    "right_distribution": distribution(right_targets),
                    "weighted_gini": round(float(weighted_gini), 6),
                }
            )

        candidates.sort(
            key=lambda item: item["weighted_gini"]
        )

        explorer[str(node_id)] = {
            "node_id": int(node_id),
            "feature": feature,
            "current_threshold": float(tree.threshold[node_id]),
            "samples": int(len(row_indices)),
            "current_gini": round(
                float(tree.impurity[node_id]),
                6,
            ),
            "candidates": candidates[:10],
        }

    return explorer

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target: str = "target",
    max_depth: int = 4,
    min_samples_leaf: int = 1,
    ccp_alpha: float = 0.0):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    contents = await file.read()

    try:
        df = pd.read_csv(BytesIO(contents))
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {error}",
        )

    if target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target '{target}' not found. "
                f"Available columns: {list(df.columns)}"
            ),
        )

    df = df.dropna()

    if len(df) < 5:
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain at least 5 rows.",
        )

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Encode categorical feature columns
    feature_encoders = {}

    for column in X.columns:

        if X[column].dtype == "object":

            encoder = LabelEncoder()

            X[column] = encoder.fit_transform(
                X[column].astype(str)
            )

            feature_encoders[column] = encoder

        else:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

    # Remove invalid feature rows
    valid = ~X.isna().any(axis=1)

    X = X.loc[valid]
    y = y.loc[valid]

    # Encode target
    target_encoder = None

    # Handle categorical/string targets robustly across pandas versions
    if pd.api.types.is_numeric_dtype(y):

        y_numeric = pd.to_numeric(
            y,
            errors="coerce"
        )

        valid_target = ~y_numeric.isna()

        X = X.loc[valid_target].copy()
        y = y_numeric.loc[valid_target]

        y = y.astype(int).to_numpy()

    else:

        y_clean = y.astype("string").str.strip()

        valid_target = (
            y_clean.notna()
            & y_clean.ne("")
        )

        X = X.loc[valid_target].copy()
        y_clean = y_clean.loc[valid_target]

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(
            y_clean.astype(str)
        )

    feature_names = list(X.columns)

    unique_classes = np.unique(y)

    if len(unique_classes) < 2:
        raise HTTPException(
            status_code=400,
            detail="Target must contain at least two classes.",
        )

    # Train/test split
    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

    except ValueError:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    # Decision Tree
    model = DecisionTreeClassifier(
        max_depth=max(1, min(int(max_depth), 20)),
        min_samples_leaf=max(1, min(int(min_samples_leaf), 10)),
        ccp_alpha=max(0.0, min(float(ccp_alpha), 0.1)),
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Save the latest trained model for /predict
    MODEL_STATE["model"] = model
    MODEL_STATE["feature_names"] = feature_names
    MODEL_STATE["feature_encoders"] = feature_encoders
    MODEL_STATE["target_encoder"] = target_encoder

    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # Class names
    if target_encoder is not None:

        class_names = [
            str(x)
            for x in target_encoder.classes_
        ]

    else:

        class_names = [
            str(x)
            for x in sorted(unique_classes)
        ]

    # Save class names for /predict
    MODEL_STATE["class_names"] = class_names

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(class_names))),
    )

    # Keep the test-row details so the confusion matrix
    # can later link directly to the affected dataset rows.
    confusion_rows = []

    for row_index, actual_class, predicted_class in zip(
        X_test.index,
        y_test,
        predictions,
    ):
        actual_class = int(actual_class)
        predicted_class = int(predicted_class)

        confusion_rows.append(
            {
                "row_index": int(row_index),
                "actual": actual_class,
                "predicted": predicted_class,
                "actual_name": class_names[actual_class],
                "predicted_name": class_names[predicted_class],
                "row": df.loc[row_index].to_dict(),
            }
        )

    # Extract tree
    tree = model.tree_

    nodes = []

    # Build parent and depth information for explainable prediction paths
    parent_map = {
        0: -1
    }

    depth_map = {
        0: 0
    }

    for parent_id in range(tree.node_count):

        left_child = int(tree.children_left[parent_id])
        right_child = int(tree.children_right[parent_id])

        if left_child != -1:
            parent_map[left_child] = parent_id
            depth_map[left_child] = depth_map[parent_id] + 1

        if right_child != -1:
            parent_map[right_child] = parent_id
            depth_map[right_child] = depth_map[parent_id] + 1

    for node_id in range(tree.node_count):

        feature_index = tree.feature[node_id]

        is_leaf = feature_index == -2

        if is_leaf:

            feature = None
            threshold = None

        else:

            feature = feature_names[feature_index]

            threshold = float(
                tree.threshold[node_id]
            )

        class_values = tree.value[node_id][0]

        predicted_class = int(
            np.argmax(class_values)
        )

        class_distribution = [
            int(round(
                float(value)
                * int(tree.n_node_samples[node_id])
            ))
            for value in class_values
        ]

        nodes.append(
            {
                "id": int(node_id),

                "parent_id": int(parent_map.get(node_id, -1)),

                "depth": int(depth_map.get(node_id, 0)),

                "feature": feature,

                "threshold": threshold,

                "samples": int(
                    tree.n_node_samples[node_id]
                ),

                "gini": round(
                    float(tree.impurity[node_id]),
                    4,
                ),

                "left": int(
                    tree.children_left[node_id]
                ),

                "right": int(
                    tree.children_right[node_id]
                ),

                "is_leaf": bool(is_leaf),

                "prediction": predicted_class,

                "prediction_name": (
                    class_names[predicted_class]
                ),

                "class_distribution": class_distribution,
            }
        )

    # Feature importance
    feature_importance = []

    for feature, value in zip(
        feature_names,
        model.feature_importances_,
    ):

        feature_importance.append(
            {
                "feature": feature,
                "importance": round(
                    float(value),
                    6,
                ),
            }
        )

    feature_importance.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    split_explorer = build_split_explorer(
        model,
        X_train,
        y_train,
        feature_names,
        class_names,
    )


    train_predictions = model.predict(X_train)

    train_accuracy = float(
        accuracy_score(y_train, train_predictions)
    )

    validation_accuracy = float(accuracy)

    actual_depth = int(model.get_depth())
    node_count = int(model.tree_.node_count)
    leaf_count = int(model.get_n_leaves())

    test_decision_path = model.decision_path(X_test)
    error_node_counts = {}

    for row_position, (actual, predicted) in enumerate(
        zip(y_test, predictions)
    ):
        if actual != predicted:
            visited_nodes = test_decision_path[row_position].indices

            for node_id in visited_nodes:
                key = str(int(node_id))
                error_node_counts[key] = (
                    error_node_counts.get(key, 0) + 1
                )

    max_error_node_count = max(
        error_node_counts.values(),
        default=0,
    )

    return {
        "dataset": {
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "features": feature_names,
            "target": target,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "rows_data": df.to_dict(orient="records"),
        },

        "metrics": {
            "accuracy": round(
                float(accuracy),
                4,
            ),
            "precision": round(
                float(precision),
                4,
            ),
            "recall": round(
                float(recall),
                4,
            ),
            "f1": round(
                float(f1),
                4,
            ),
        },

        "confusion_matrix": cm.tolist(),

        "confusion_rows": confusion_rows,

        "class_names": class_names,

        "split_explorer": split_explorer,

        "tree": nodes,

        "feature_importance": feature_importance,

        "complexity": {
            "requested_max_depth": int(max_depth),
            "requested_min_samples_leaf": int(min_samples_leaf),
            "requested_ccp_alpha": float(ccp_alpha),
            "actual_depth": actual_depth,
            "node_count": node_count,
            "leaf_count": leaf_count,
            "train_accuracy": round(train_accuracy, 4),
            "validation_accuracy": round(validation_accuracy, 4),
            "generalization_gap": round(
                train_accuracy - validation_accuracy,
                4,
            ),
            "error_node_counts": error_node_counts,
            "max_error_node_count": max_error_node_count,
        },
    }


# ============================================================
# P3 — MODEL UNDERSTANDING LAB
# ============================================================

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA
import numpy as np


def p3_prepare_dataset(file_bytes, target):
    import io

    df = pd.read_csv(io.BytesIO(file_bytes))

    if target not in df.columns:
        raise ValueError(
            f"Target '{target}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[target]).copy()

    X = df.drop(columns=[target])
    y = df[target].copy()

    feature_names = list(X.columns)

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    X_encoded = X.copy()

    feature_encoders = {}

    for column in categorical_columns:
        encoder = LabelEncoder()

        values = X_encoded[column].astype(str)

        X_encoded[column] = encoder.fit_transform(values)

        feature_encoders[column] = encoder

    X_encoded = X_encoded.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for column in X_encoded.columns:
        X_encoded[column] = pd.to_numeric(
            X_encoded[column],
            errors="coerce"
        )

        X_encoded[column] = X_encoded[column].fillna(
            X_encoded[column].median()
        )

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(
        y.astype(str)
    )

    class_names = [
        str(value)
        for value in target_encoder.classes_
    ]

    if len(class_names) < 2:
        raise ValueError(
            "The target must contain at least two classes."
        )

    return (
        df,
        X_encoded.astype(float),
        y_encoded,
        feature_names,
        class_names,
    )


def p3_metrics(y_true, predictions):
    return {
        "accuracy": round(
            float(
                accuracy_score(
                    y_true,
                    predictions
                )
            ),
            4,
        ),
        "precision": round(
            float(
                precision_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
    }


@app.post("/p3/analyze")
async def p3_analyze(
    file: UploadFile = File(...),
    target: str = "target",
    model_name: str = "logistic",
):
    try:
        contents = await file.read()

        (
            df,
            X,
            y,
            feature_names,
            class_names,
        ) = p3_prepare_dataset(
            contents,
            target,
        )

        if len(X) < 6:
            raise ValueError(
                "Dataset needs at least 6 rows for P3 analysis."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        model_key = model_name.lower().strip()

        if model_key == "logistic":
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
            )

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

            coefficients = []

            raw_coef = model.coef_

            if raw_coef.ndim == 1:
                raw_coef = raw_coef.reshape(1, -1)

            importance_values = np.mean(
                np.abs(raw_coef),
                axis=0
            )

            for feature, value in zip(
                feature_names,
                importance_values
            ):
                coefficients.append({
                    "feature": str(feature),
                    "value": round(
                        float(value),
                        6
                    ),
                })

            coefficients.sort(
                key=lambda item: item["value"],
                reverse=True,
            )

        elif model_key == "knn":
            model = KNeighborsClassifier(
                n_neighbors=min(
                    5,
                    max(
                        1,
                        len(X_train) - 1
                    ),
                )
            )

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

            coefficients = []

        elif model_key == "svm":
            model = SVC(
                kernel="rbf",
                probability=True,
                random_state=42,
            )

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

            coefficients = []

        elif model_key == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            coefficients = [
                {
                    "feature": str(feature),
                    "value": round(
                        float(value),
                        6
                    ),
                }
                for feature, value in zip(
                    feature_names,
                    model.feature_importances_,
                )
            ]

            coefficients.sort(
                key=lambda item: item["value"],
                reverse=True,
            )

        elif model_key == "decision_tree":
            model = DecisionTreeClassifier(
                max_depth=4,
                random_state=42,
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            coefficients = [
                {
                    "feature": str(feature),
                    "value": round(
                        float(value),
                        6
                    ),
                }
                for feature, value in zip(
                    feature_names,
                    model.feature_importances_,
                )
            ]

            coefficients.sort(
                key=lambda item: item["value"],
                reverse=True,
            )

        else:
            raise ValueError(
                "Unsupported P3 model. "
                "Use decision_tree, logistic, knn, svm, "
                "or random_forest."
            )

        metrics = p3_metrics(
            y_test,
            predictions
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
        ).tolist()

        probability_rows = []

        if hasattr(
            model,
            "predict_proba"
        ):
            probabilities = model.predict_proba(
                X_test_scaled
                if model_key in {
                    "logistic",
                    "knn",
                    "svm",
                }
                else X_test
            )

            for index, row in enumerate(
                probabilities
            ):
                probability_rows.append({
                    "row": int(
                        X_test.index[index]
                    ),
                    "actual": class_names[
                        int(y_test[index])
                    ],
                    "predicted": class_names[
                        int(predictions[index])
                    ],
                    "probabilities": {
                        class_names[class_index]:
                            round(
                                float(value),
                                4
                            )
                        for class_index, value
                        in enumerate(row)
                    },
                })

        # ----------------------------------------------------
        # 2D projection for visualization.
        # This is intentionally a visualization projection,
        # not a replacement for the full-dimensional model.
        # ----------------------------------------------------

        projection_scaler = StandardScaler()

        X_scaled_all = projection_scaler.fit_transform(X)

        if X.shape[1] >= 2:
            pca = PCA(
                n_components=2,
                random_state=42
            )

            projected = pca.fit_transform(
                X_scaled_all
            )

            projection_explained = [
                round(
                    float(value),
                    4
                )
                for value in pca.explained_variance_ratio_
            ]
        else:
            first = X_scaled_all[:, 0]

            projected = np.column_stack([
                first,
                np.zeros_like(first)
            ])

            projection_explained = [
                1.0,
                0.0,
            ]

        boundary_model = SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
        )

        boundary_model.fit(
            projected,
            y
        )

        min_x = float(
            projected[:, 0].min()
        )

        max_x = float(
            projected[:, 0].max()
        )

        min_y = float(
            projected[:, 1].min()
        )

        max_y = float(
            projected[:, 1].max()
        )

        x_padding = max(
            0.5,
            (max_x - min_x) * 0.12
        )

        y_padding = max(
            0.5,
            (max_y - min_y) * 0.12
        )

        grid_x = np.linspace(
            min_x - x_padding,
            max_x + x_padding,
            35,
        )

        grid_y = np.linspace(
            min_y - y_padding,
            max_y + y_padding,
            35,
        )

        grid = np.array([
            [x, y_value]
            for x in grid_x
            for y_value in grid_y
        ])

        grid_prediction = boundary_model.predict(
            grid
        )

        boundary_points = []

        for point, prediction in zip(
            grid,
            grid_prediction
        ):
            boundary_points.append({
                "x": round(
                    float(point[0]),
                    4
                ),
                "y": round(
                    float(point[1]),
                    4
                ),
                "class_index": int(
                    prediction
                ),
            })

        scatter_rows = []

        for index, point in enumerate(
            projected
        ):
            scatter_rows.append({
                "row": int(
                    df.index[index]
                ),
                "x": round(
                    float(point[0]),
                    4
                ),
                "y": round(
                    float(point[1]),
                    4
                ),
                "class_index": int(
                    y[index]
                ),
                "class_name": class_names[
                    int(y[index])
                ],
            })

        # ----------------------------------------------------
        # KNN neighbor explanation
        # ----------------------------------------------------

        neighbors = []

        if model_key == "knn":
            distances, indices = model.kneighbors(
                X_test_scaled,
                n_neighbors=min(
                    5,
                    len(X_train)
                ),
            )

            for test_position in range(
                len(X_test)
            ):
                neighbor_items = []

                for distance, train_position in zip(
                    distances[test_position],
                    indices[test_position],
                ):
                    neighbor_items.append({
                        "row": int(
                            X_train.index[
                                train_position
                            ]
                        ),
                        "distance": round(
                            float(distance),
                            4
                        ),
                        "class": class_names[
                            int(
                                y_train[
                                    train_position
                                ]
                            )
                        ],
                    })

                neighbors.append({
                    "test_row": int(
                        X_test.index[
                            test_position
                        ]
                    ),
                    "neighbors": neighbor_items,
                })

        svm_info = None

        if model_key == "svm":
            svm_info = {
                "kernel": "rbf",
                "support_vector_count": int(
                    len(model.support_)
                ),
                "support_vectors_per_class": [
                    int(value)
                    for value in model.n_support_
                ],
            }

        forest_info = None

        if model_key == "random_forest":
            tree_count = len(
                model.estimators_
            )

            forest_predictions = []

            for estimator in model.estimators_:
                forest_predictions.append(
                    estimator.predict(
                        X_test
                    )
                )

            forest_predictions = np.array(
                forest_predictions
            )

            voting = []

            for row_position in range(
                len(X_test)
            ):
                votes = {}

                for prediction in forest_predictions[
                    :, row_position
                ]:
                    key = int(prediction)
                    votes[key] = (
                        votes.get(key, 0) + 1
                    )

                voting.append({
                    "row": int(
                        X_test.index[row_position]
                    ),
                    "votes": {
                        class_names[int(key)]:
                            int(value)
                        for key, value in votes.items()
                    },
                    "trees": tree_count,
                })

            forest_info = {
                "tree_count": tree_count,
                "voting": voting,
            }

        return {
            "model": model_key,
            "classes": class_names,
            "features": feature_names,
            "metrics": metrics,
            "confusion_matrix": matrix,
            "feature_effects": coefficients,
            "probabilities": probability_rows,
            "projection": {
                "explained_variance": projection_explained,
                "points": scatter_rows,
                "boundary": boundary_points,
            },
            "knn_neighbors": neighbors,
            "svm": svm_info,
            "random_forest": forest_info,
            "dataset": {
                "rows": int(len(df)),
                "features": int(len(feature_names)),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"P3 analysis failed: {str(exc)}"
        )
