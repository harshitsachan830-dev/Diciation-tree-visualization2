"""
Test suite for Model Mind Backend API and ML services.
"""

import io
import sys
import os
import unittest
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app
from backend.data_service import (
    load_default_datasets, DATASET_METADATA,
    preprocess_dataset, get_dataset
)
from backend.model_service import (
    train_decision_tree, get_run, predict_row,
    get_2d_boundary_data, get_3d_feature_space_data, extract_rules
)

class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        load_default_datasets()

    def test_01_health_check(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")
        print("✓ Health check passed")

    def test_02_datasets_listing(self):
        res = self.client.get("/api/datasets")
        self.assertEqual(res.status_code, 200)
        datasets = res.json()
        names = [d["name"] for d in datasets]
        self.assertIn("iris", names)
        self.assertIn("titanic", names)
        self.assertIn("housing", names)
        print(f"✓ Default datasets loaded: {names}")

    def test_03_missing_value_imputation(self):
        # Find titanic dataset
        res = self.client.get("/api/datasets")
        titanic = next(d for d in res.json() if d["name"] == "titanic")
        self.assertTrue(titanic["missing_summary"]["has_missing"])
        
        # Test Mean Imputation
        res_impute = self.client.post("/api/datasets/preprocess", json={
            "dataset_id": titanic["dataset_id"],
            "impute_strategy": "mean"
        })
        self.assertEqual(res_impute.status_code, 200)
        data = res_impute.json()
        self.assertFalse(data["missing_summary"]["has_missing"])
        print("✓ Mean imputation resolved all missing values")

    def test_04_feature_scaling(self):
        res = self.client.get("/api/datasets")
        iris = next(d for d in res.json() if d["name"] == "iris")
        
        # Apply StandardScaler
        res_scale = self.client.post("/api/datasets/preprocess", json={
            "dataset_id": iris["dataset_id"],
            "scale_method": "standard"
        })
        self.assertEqual(res_scale.status_code, 200)
        print("✓ StandardScaler applied successfully")

    def test_05_train_decision_tree(self):
        res = self.client.get("/api/datasets")
        iris = next(d for d in res.json() if d["name"] == "iris")
        
        train_res = self.client.post("/api/models/decision-tree/train", json={
            "dataset_id": iris["dataset_id"],
            "target": "species",
            "task": "classification",
            "criterion": "gini",
            "max_depth": 4,
            "min_samples_split": 2
        })
        self.assertEqual(train_res.status_code, 200)
        run_data = train_res.json()
        self.assertIn("run_id", run_data)
        self.assertGreater(run_data["metrics"]["test_accuracy"], 0.8)
        print(f"✓ Decision tree trained: Acc={run_data['metrics']['test_accuracy']}, Nodes={run_data['total_nodes']}")

        # Test Tree graph retrieval
        run_id = run_data["run_id"]
        tree_res = self.client.get(f"/api/runs/{run_id}/tree")
        self.assertEqual(tree_res.status_code, 200)
        tree = tree_res.json()
        self.assertGreater(len(tree["nodes"]), 0)
        print(f"✓ Retrieved tree structure with {len(tree['nodes'])} nodes and {len(tree['edges'])} edges")

        # Test 12-field node inspector
        node_res = self.client.get(f"/api/runs/{run_id}/nodes/0")
        self.assertEqual(node_res.status_code, 200)
        node0 = node_res.json()
        self.assertEqual(node0["node_id"], 0)
        self.assertIn("impurity", node0)
        self.assertIn("samples", node0)
        print(f"✓ Root node inspector: split on {node0.get('feature')}")

        # Test Prediction Explorer & Path Tracing
        pred_res = self.client.post(f"/api/runs/{run_id}/predict", json={
            "features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        })
        self.assertEqual(pred_res.status_code, 200)
        pred_data = pred_res.json()
        self.assertIn(pred_data["prediction"], ["setosa", "versicolor", "virginica"])
        self.assertGreater(len(pred_data["path_steps"]), 0)
        print(f"✓ Prediction: {pred_data['prediction']} with {len(pred_data['path_steps'])} breadcrumb steps")

        # Test 2D Decision boundary
        b2d_res = self.client.get(f"/api/runs/{run_id}/boundary-2d")
        self.assertEqual(b2d_res.status_code, 200)
        b2d = b2d_res.json()
        self.assertIn("grid_z", b2d)
        print(f"✓ 2D Boundary generated: {b2d['feature_x']} vs {b2d['feature_y']}")

        # Test 3D Feature Space
        s3d_res = self.client.get(f"/api/runs/{run_id}/feature-space-3d")
        self.assertEqual(s3d_res.status_code, 200)
        s3d = s3d_res.json()
        self.assertIn("points", s3d)
        print(f"✓ 3D Feature Space generated with {len(s3d['points'])} points")

        # Test Rules
        rules_res = self.client.get(f"/api/runs/{run_id}/rules")
        self.assertEqual(rules_res.status_code, 200)
        rules = rules_res.json()
        self.assertGreater(len(rules), 0)
        print(f"✓ Generated {len(rules)} human-readable IF-THEN rules")

    def test_06_csv_file_upload(self):
        csv_content = b"feat1,feat2,target\n1.0,2.0,cat\n1.5,,cat\n4.0,5.0,dog\n"
        res = self.client.post(
            "/api/datasets",
            files={"file": ("test_upload.csv", io.BytesIO(csv_content), "text/csv")}
        )
        self.assertEqual(res.status_code, 200)
        meta = res.json()
        self.assertEqual(meta["rows"], 3)
        self.assertTrue(meta["missing_summary"]["has_missing"])
        print("✓ CSV Upload API tested successfully")

if __name__ == "__main__":
    unittest.main()
