"""
Model Mind - Decision Tree Studio Backend API
FastAPI implementation adhering to Section 14 & 15 API contracts.
Supports CSV uploads, missing value imputation (Mean/Median/Mode),
feature scaling (Standard/MinMax/Robust), decision tree training,
2D & 3D visualizations, prediction tracing, and evaluation metrics.
"""

import io
import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from .data_service import (
    register_dataset, get_dataset, preprocess_dataset,
    DATASET_METADATA, load_default_datasets
)
from .model_service import (
    train_decision_tree, get_run, predict_row,
    get_2d_boundary_data, get_3d_feature_space_data, extract_rules
)

app = FastAPI(
    title="Model Mind Decision Tree Studio API",
    description="Backend API supporting the Decision Tree Visualization specification",
    version="1.0.0"
)

# Enable CORS for browser client requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load default datasets on startup
@app.on_event("startup")
def startup_event():
    load_default_datasets()

# ----------------- Request Models -----------------
class PreprocessRequest(BaseModel):
    dataset_id: str
    impute_strategy: Optional[str] = None # 'mean', 'median', 'mode', 'drop'
    impute_columns: Optional[List[str]] = None
    scale_method: Optional[str] = None    # 'standard', 'minmax', 'robust'
    scale_columns: Optional[List[str]] = None

class TrainRequest(BaseModel):
    dataset_id: str
    target: str
    task: str = "classification" # 'classification' or 'regression'
    selected_features: Optional[List[str]] = None
    criterion: str = "gini"
    max_depth: Optional[int] = 5
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    ccp_alpha: float = 0.0
    test_size: float = 0.2
    random_state: int = 42

class PredictRequest(BaseModel):
    features: Dict[str, Any]

# ----------------- API Endpoints -----------------

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Model Mind Decision Tree Studio",
        "datasets_loaded": list(DATASET_METADATA.keys())
    }

@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload CSV dataset, inspect columns, infer types, calculate missing value summary."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        filename = file.filename or "uploaded.csv"
        clean_name = os.path.splitext(filename)[0]
        meta = register_dataset(df, name=clean_name)
        return meta
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

@app.get("/api/datasets")
def list_datasets():
    """List all registered datasets."""
    return list(DATASET_METADATA.values())

@app.get("/api/datasets/{dataset_id}/preview")
def get_dataset_preview(dataset_id: str, limit: int = 20):
    """Retrieve preview rows and column metadata for a dataset."""
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    meta = DATASET_METADATA.get(dataset_id, {})
    return {
        "dataset_id": dataset_id,
        "name": meta.get("name"),
        "rows": len(df),
        "columns": list(df.columns),
        "inferred_types": meta.get("inferred_types"),
        "missing_summary": meta.get("missing_summary"),
        "preview": df.head(limit).replace({float('nan'): None}).to_dict(orient="records")
    }

@app.post("/api/datasets/preprocess")
def preprocess_data(req: PreprocessRequest):
    """Perform missing value imputation (Mean, Median, Mode) and Feature Scaling."""
    try:
        res = preprocess_dataset(
            dataset_id=req.dataset_id,
            impute_strategy=req.impute_strategy,
            impute_columns=req.impute_columns,
            scale_method=req.scale_method,
            scale_columns=req.scale_columns
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/models/decision-tree/train")
def train_tree(req: TrainRequest):
    """Train Decision Tree and evaluate model performance."""
    df = get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        res = train_decision_tree(
            df=df,
            target_col=req.target,
            feature_cols=req.selected_features,
            task=req.task,
            criterion=req.criterion,
            max_depth=req.max_depth,
            min_samples_split=req.min_samples_split,
            min_samples_leaf=req.min_samples_leaf,
            ccp_alpha=req.ccp_alpha,
            test_size=req.test_size,
            random_state=req.random_state
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {str(e)}")

@app.get("/api/runs/{run_id}/tree")
def get_tree_graph(run_id: str):
    """Return tree graph metadata: nodes, edges, class names, feature names."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run_id,
        "task": run["task"],
        "nodes": run["nodes"],
        "edges": run["edges"],
        "class_names": run["class_names"],
        "encoded_features": run["encoded_features"],
        "total_nodes": run["total_nodes"],
        "leaves_count": run["leaves_count"],
        "max_depth": run["max_depth"]
    }

@app.get("/api/runs/{run_id}/nodes/{node_id}")
def get_node_details(run_id: str, node_id: int):
    """Retrieve 12-field node details for inspector panel."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    for n in run["nodes"]:
        if n["node_id"] == node_id:
            return n
    raise HTTPException(status_code=404, detail="Node not found")

@app.post("/api/runs/{run_id}/predict")
def predict_sample(run_id: str, req: PredictRequest):
    """Predict one row with full decision path breadcrumbs."""
    try:
        return predict_row(run_id, req.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/runs/{run_id}/metrics")
def get_metrics(run_id: str):
    """Get task-appropriate evaluation metrics."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run["metrics"]

@app.get("/api/runs/{run_id}/importance")
def get_importance(run_id: str):
    """Get feature importance scores."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run["feature_importances"]

@app.get("/api/runs/{run_id}/boundary-2d")
def get_2d_boundary(run_id: str, feat_x: Optional[str] = None, feat_y: Optional[str] = None):
    """Return 2D decision boundary grid and sample points."""
    try:
        return get_2d_boundary_data(run_id, feat_x, feat_y)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/runs/{run_id}/feature-space-3d")
def get_3d_feature_space(run_id: str, feat_x: Optional[str] = None, feat_y: Optional[str] = None, feat_z: Optional[str] = None):
    """Return 3D feature space coordinates and target values."""
    try:
        return get_3d_feature_space_data(run_id, feat_x, feat_y, feat_z)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/runs/{run_id}/rules")
def get_rules(run_id: str):
    """Return human-readable IF-THEN rules."""
    try:
        return extract_rules(run_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/report")
def get_prd_report():
    """Return the complete 32-section implementation report specification."""
    report_path = os.path.join(os.path.dirname(__file__), "report_sections.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)
    return []

# Mount static frontend directory
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    """Serve the main frontend application."""
    frontend_index = os.path.join(frontend_dir, "index.html")
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    # Fallback to project root index.html
    root_index = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.html"))
    if os.path.exists(root_index):
        return FileResponse(root_index)
    return {"message": "Model Mind Decision Tree Studio API running"}
