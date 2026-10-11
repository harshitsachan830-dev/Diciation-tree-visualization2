"""
Model Mind - Decision Tree Studio Model Service
Trains scikit-learn DecisionTreeClassifier and DecisionTreeRegressor, extracts complete
tree node hierarchy, calculates evaluation metrics, generates 2D decision boundary grids,
3D feature space coordinates, traces prediction paths, and outputs IF-THEN rules.
"""

import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, r2_score,
    mean_absolute_error, mean_squared_error
)

RUNS: Dict[str, Dict[str, Any]] = {}

def train_decision_tree(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    task: str = "classification", # 'classification' or 'regression'
    criterion: str = "gini",       # 'gini', 'entropy', 'log_loss', 'squared_error', 'absolute_error'
    max_depth: Optional[int] = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    ccp_alpha: float = 0.0,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())[:8]
    
    # 1. Prepare data
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
        
    X_raw = df[feature_cols].copy()
    y_raw = df[target_col].copy()
    
    # Impute any remaining missing values
    for col in X_raw.columns:
        if X_raw[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_raw[col]):
                X_raw[col] = X_raw[col].fillna(X_raw[col].median())
            else:
                X_raw[col] = X_raw[col].fillna(X_raw[col].mode().iloc[0] if not X_raw[col].mode().empty else "missing")
                
    # One-hot encode categoricals for scikit-learn
    X_encoded = pd.get_dummies(X_raw, drop_first=False)
    encoded_features = list(X_encoded.columns)
    
    # Handle Target
    is_classification = (task == "classification")
    class_names = []
    
    if is_classification:
        if y_raw.isnull().any():
            y_raw = y_raw.fillna(y_raw.mode().iloc[0])
        # Convert classes to strings
        y_encoded = y_raw.astype(str)
        class_names = sorted(list(y_encoded.unique()))
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        y_numeric = y_encoded.map(class_to_idx).values
        
        # Scikit-learn criterion mapping
        valid_crit = criterion if criterion in ["gini", "entropy", "log_loss"] else "gini"
        model = DecisionTreeClassifier(
            criterion=valid_crit,
            max_depth=max_depth if max_depth and max_depth > 0 else None,
            min_samples_split=max(2, min_samples_split),
            min_samples_leaf=max(1, min_samples_leaf),
            ccp_alpha=max(0.0, ccp_alpha),
            random_state=random_state
        )
    else:
        # Regression
        y_numeric = pd.to_numeric(y_raw, errors="coerce").fillna(y_raw.mean()).values
        valid_crit = criterion if criterion in ["squared_error", "friedman_mse", "absolute_error", "poisson"] else "squared_error"
        model = DecisionTreeRegressor(
            criterion=valid_crit,
            max_depth=max_depth if max_depth and max_depth > 0 else None,
            min_samples_split=max(2, min_samples_split),
            min_samples_leaf=max(1, min_samples_leaf),
            ccp_alpha=max(0.0, ccp_alpha),
            random_state=random_state
        )

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded.values, y_numeric, test_size=test_size, random_state=random_state
    )
    
    # 3. Fit
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 4. Compute Metrics
    metrics: Dict[str, Any] = {}
    if is_classification:
        acc_train = float(accuracy_score(y_train, y_pred_train))
        acc_test = float(accuracy_score(y_test, y_pred_test))
        prec = float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred_test).tolist()
        cr = classification_report(y_test, y_pred_test, target_names=class_names, output_dict=True, zero_division=0)
        
        metrics = {
            "task": "classification",
            "train_accuracy": round(acc_train, 4),
            "test_accuracy": round(acc_test, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "classification_report": cr,
            "class_names": class_names
        }
    else:
        r2_test = float(r2_score(y_test, y_pred_test))
        mae = float(mean_absolute_error(y_test, y_pred_test))
        mse = float(mean_squared_error(y_test, y_pred_test))
        rmse = float(np.sqrt(mse))
        
        residuals = (y_test - y_pred_test).tolist()
        metrics = {
            "task": "regression",
            "r2_score": round(r2_test, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "actual_values": y_test.tolist()[:50],
            "predicted_values": y_pred_test.tolist()[:50],
            "residuals": residuals[:50]
        }

    # 5. Extract Tree Structure (Section 15.1 format)
    tree_ = model.tree_
    nodes = []
    edges = []
    total_train_samples = len(X_train)
    
    def recurse(node_id: int, depth: int, parent_id: Optional[int] = None, condition: str = ""):
        is_leaf = (tree_.children_left[node_id] == _tree.TREE_LEAF)
        samples = int(tree_.n_node_samples[node_id])
        impurity = float(tree_.impurity[node_id])
        val = tree_.value[node_id][0]
        coverage_pct = round((samples / total_train_samples) * 100, 2)
        
        if is_leaf:
            if is_classification:
                pred_idx = int(np.argmax(val))
                pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                dist = {class_names[i]: int(val[i]) for i in range(len(class_names))}
            else:
                pred_label = round(float(val[0]), 3)
                dist = {"mean": round(float(val[0]), 3)}
                
            node_data = {
                "node_id": node_id,
                "depth": depth,
                "kind": "leaf",
                "impurity": round(impurity, 4),
                "samples": samples,
                "coverage_pct": coverage_pct,
                "value": [int(x) for x in val] if is_classification else [round(float(val[0]), 3)],
                "prediction": pred_label,
                "distribution": dist,
                "left_child": None,
                "right_child": None
            }
            nodes.append(node_data)
        else:
            feature_idx = int(tree_.feature[node_id])
            feature_name = encoded_features[feature_idx]
            threshold = round(float(tree_.threshold[node_id]), 4)
            left_child = int(tree_.children_left[node_id])
            right_child = int(tree_.children_right[node_id])
            
            node_data = {
                "node_id": node_id,
                "depth": depth,
                "kind": "split",
                "feature": feature_name,
                "threshold": threshold,
                "impurity": round(impurity, 4),
                "samples": samples,
                "coverage_pct": coverage_pct,
                "value": [int(x) for x in val] if is_classification else [round(float(val[0]), 3)],
                "left_child": left_child,
                "right_child": right_child
            }
            nodes.append(node_data)
            
            # Recurse children
            recurse(left_child, depth + 1, node_id, f"≤ {threshold}")
            recurse(right_child, depth + 1, node_id, f"> {threshold}")
            
        if parent_id is not None:
            edges.append({
                "source": parent_id,
                "target": node_id,
                "condition": condition
            })
            
    recurse(0, 0)
    
    # Feature importances
    importances = {
        encoded_features[i]: round(float(model.feature_importances_[i]), 4)
        for i in range(len(encoded_features))
    }
    # Sort descending
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    # Store Run
    run_record = {
        "run_id": run_id,
        "task": task,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "encoded_features": encoded_features,
        "class_names": class_names,
        "model": model,
        "nodes": nodes,
        "edges": edges,
        "metrics": metrics,
        "feature_importances": sorted_importances,
        "total_nodes": len(nodes),
        "leaves_count": len([n for n in nodes if n["kind"] == "leaf"]),
        "max_depth": int(model.get_depth()),
        "df_processed": X_encoded,
        "y_numeric": y_numeric
    }
    RUNS[run_id] = run_record
    
    return {
        "run_id": run_id,
        "total_nodes": run_record["total_nodes"],
        "leaves_count": run_record["leaves_count"],
        "max_depth": run_record["max_depth"],
        "metrics": metrics,
        "feature_importances": sorted_importances
    }

def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    return RUNS.get(run_id)

def predict_row(run_id: str, row_dict: Dict[str, Any]) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")
        
    model = run["model"]
    encoded_features = run["encoded_features"]
    
    # Build vector
    row_df = pd.DataFrame([row_dict])
    encoded_row = pd.get_dummies(row_df).reindex(columns=encoded_features, fill_value=0)
    vec = encoded_row.values
    
    # Predict
    pred = model.predict(vec)[0]
    probs = None
    if run["task"] == "classification" and hasattr(model, "predict_proba"):
        probs = {
            run["class_names"][i]: round(float(p), 4)
            for i, p in enumerate(model.predict_proba(vec)[0])
        }
        pred_label = run["class_names"][int(pred)] if int(pred) < len(run["class_names"]) else str(pred)
    else:
        pred_label = round(float(pred), 4)
        
    # Decision path tracing
    tree_ = model.tree_
    node_indicator = model.decision_path(vec)
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    path_steps = []
    for node_id in node_index:
        is_leaf = (tree_.children_left[node_id] == _tree.TREE_LEAF)
        if not is_leaf:
            feat_idx = tree_.feature[node_id]
            feat_name = encoded_features[feat_idx]
            threshold = tree_.threshold[node_id]
            actual_val = vec[0, feat_idx]
            passed = actual_val <= threshold
            path_steps.append({
                "node_id": int(node_id),
                "feature": feat_name,
                "operator": "≤" if passed else ">",
                "threshold": round(float(threshold), 3),
                "value": round(float(actual_val), 3),
                "branch": "left" if passed else "right"
            })
        else:
            path_steps.append({
                "node_id": int(node_id),
                "leaf": True,
                "samples": int(tree_.n_node_samples[node_id])
            })
            
    return {
        "prediction": pred_label,
        "probabilities": probs,
        "leaf_id": int(node_index[-1]),
        "path_node_ids": [int(x) for x in node_index],
        "path_steps": path_steps
    }

def get_2d_boundary_data(run_id: str, feat_x: Optional[str] = None, feat_y: Optional[str] = None) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")
        
    encoded_features = run["encoded_features"]
    if len(encoded_features) < 2:
        return {"error": "Need at least 2 features for 2D boundary"}
        
    fx = feat_x if (feat_x and feat_x in encoded_features) else encoded_features[0]
    fy = feat_y if (feat_y and feat_y in encoded_features and feat_y != fx) else encoded_features[1]
    
    X_df = run["df_processed"]
    x_min, x_max = float(X_df[fx].min()), float(X_df[fx].max())
    y_min, y_max = float(X_df[fy].min()), float(X_df[fy].max())
    
    # Margin
    dx = (x_max - x_min) * 0.1 or 1.0
    dy = (y_max - y_min) * 0.1 or 1.0
    
    grid_x = np.linspace(x_min - dx, x_max + dx, 50)
    grid_y = np.linspace(y_min - dy, y_max + dy, 50)
    xx, yy = np.meshgrid(grid_x, grid_y)
    
    # Baseline with medians
    baseline = X_df.median().to_dict()
    mesh_rows = []
    for xi, yi in zip(xx.ravel(), yy.ravel()):
        row = baseline.copy()
        row[fx] = xi
        row[fy] = yi
        mesh_rows.append(row)
        
    mesh_df = pd.DataFrame(mesh_rows)[encoded_features]
    preds = run["model"].predict(mesh_df.values)
    zz = preds.reshape(xx.shape).tolist()
    
    # Scatter samples
    scatter_pts = []
    sample_df = X_df[[fx, fy]].copy()
    targets = run["y_numeric"]
    for i in range(min(150, len(sample_df))):
        scatter_pts.append({
            "x": round(float(sample_df.iloc[i][fx]), 3),
            "y": round(float(sample_df.iloc[i][fy]), 3),
            "target": run["class_names"][int(targets[i])] if run["task"] == "classification" else round(float(targets[i]), 2)
        })
        
    return {
        "feature_x": fx,
        "feature_y": fy,
        "grid_x": grid_x.tolist(),
        "grid_y": grid_y.tolist(),
        "grid_z": zz,
        "scatter_points": scatter_pts,
        "class_names": run["class_names"]
    }

def get_3d_feature_space_data(run_id: str, feat_x: Optional[str] = None, feat_y: Optional[str] = None, feat_z: Optional[str] = None) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")
        
    encoded_features = run["encoded_features"]
    if len(encoded_features) < 3:
        return {"error": "Need at least 3 features for 3D visualization"}
        
    fx = feat_x or encoded_features[0]
    fy = feat_y or encoded_features[1]
    fz = feat_z or encoded_features[2]
    
    X_df = run["df_processed"]
    pts = []
    targets = run["y_numeric"]
    for i in range(min(200, len(X_df))):
        pts.append({
            "x": round(float(X_df.iloc[i][fx]), 3),
            "y": round(float(X_df.iloc[i][fy]), 3),
            "z": round(float(X_df.iloc[i][fz]), 3),
            "target": run["class_names"][int(targets[i])] if run["task"] == "classification" else round(float(targets[i]), 2)
        })
        
    return {
        "feature_x": fx,
        "feature_y": fy,
        "feature_z": fz,
        "points": pts,
        "class_names": run["class_names"]
    }

def extract_rules(run_id: str) -> List[Dict[str, Any]]:
    run = RUNS.get(run_id)
    if not run:
        raise ValueError(f"Run {run_id} not found")
        
    model = run["model"]
    tree_ = model.tree_
    encoded_features = run["encoded_features"]
    class_names = run["class_names"]
    total_samples = len(run["df_processed"])
    
    rules = []
    
    def recurse(node_id: int, current_rules: List[str]):
        is_leaf = (tree_.children_left[node_id] == _tree.TREE_LEAF)
        if is_leaf:
            val = tree_.value[node_id][0]
            samples = int(tree_.n_node_samples[node_id])
            if run["task"] == "classification":
                pred_idx = int(np.argmax(val))
                pred = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                purity = round((float(np.max(val)) / samples) * 100, 1) if samples > 0 else 100
            else:
                pred = round(float(val[0]), 3)
                purity = round(1.0 - float(tree_.impurity[node_id]), 2)
                
            rule_str = " AND ".join(current_rules) if current_rules else "ALL SAMPLES"
            rules.append({
                "leaf_id": node_id,
                "condition": f"IF {rule_str} THEN PREDICT {pred}",
                "samples": samples,
                "coverage_pct": round((samples / total_samples) * 100, 2),
                "purity_pct": purity,
                "prediction": pred
            })
        else:
            feat_name = encoded_features[tree_.feature[node_id]]
            thresh = round(float(tree_.threshold[node_id]), 3)
            recurse(int(tree_.children_left[node_id]), current_rules + [f"({feat_name} ≤ {thresh})"])
            recurse(int(tree_.children_right[node_id]), current_rules + [f"({feat_name} > {thresh})"])
            
    recurse(0, [])
    return rules
