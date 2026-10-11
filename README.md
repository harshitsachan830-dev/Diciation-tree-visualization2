# Model Mind — Decision Tree Visualization Studio

A full-stack interactive web platform for training, visualizing, inspecting, and evaluating Decision Trees with 2D decision boundaries, 3D feature spaces, missing value handling, and feature scaling. Built strictly according to the PRD specification report.

---

## 🚀 Key Features

1. **Interactive Decision Tree Visualizer (SVG)**
   - Pan, zoom, auto-fit (`fitTree`) layout.
   - 12-field Mathematical Node Details Inspector (Depth, Impurity, Sample Count, Coverage %, Decision Threshold, Class Distribution, Child Pointers).
   - Real-time golden glowing edge trajectory animations for sample predictions.

2. **2D & 3D Visual Graphs**
   - **2D Decision Boundaries**: High-resolution contour meshgrid showing exact decision regions with sample scatter overlay.
   - **3D Feature Space**: Interactive 3D WebGL scatter plot with rotating camera in Plotly.

3. **CSV Upload & Data Preprocessing**
   - Upload any custom CSV dataset.
   - Automatic column type inference (`numeric` vs `categorical`).
   - **Missing Value Handling**: Detects missing cells and applies **Mean**, **Median**, or **Mode** (or Drop rows) imputation.
   - **Feature Scaling**: Implements **StandardScaler** (Z-Score), **MinMaxScaler** ([0, 1]), and **RobustScaler** (IQR).

4. **Prediction Explorer**
   - Test sample prediction with step-by-step breadcrumb path table and golden path highlights on the SVG tree.

5. **Model Evaluation & Rules**
   - Confusion Matrix heatmap and Classification Report (Precision, Recall, F1).
   - Regression diagnostics: $R^2$, MAE, MSE, RMSE, Residuals scatter, Actual vs Predicted.
   - Human-readable IF-THEN decision rules with coverage and purity metrics.

6. **Full PRD Implementation Report Viewer**
   - Embedded interactive documentation reader displaying all 32 PRD sections.

---

## 📁 Project Architecture

```
/Users/harshit9793/Decition tree/
├── backend/
│   ├── main.py              # FastAPI server implementing all Section 15 API contracts
│   ├── model_service.py     # DecisionTree training, 2D/3D grids, path tracer, rules extraction
│   ├── data_service.py      # Missing values (Mean/Median/Mode) & Feature scaling
│   ├── requirements.txt     # Backend dependencies (fastapi, scikit-learn, pandas, etc.)
│   └── test_backend.py      # Automated unit test suite verifying all endpoints
├── frontend/
│   ├── index.html           # Modular frontend layout
│   ├── css/
│   │   └── style.css        # Obsidian/Charcoal theme with Emerald accents
│   └── js/
│       ├── api.js           # REST API client connecting to FastAPI
│       ├── tree_renderer.js # SVG Tree layout, zoom/pan/fit, node inspector, path tracer
│       ├── charts.js        # Plotly 2D/3D charts, confusion matrix, residuals
│       └── app.js           # State orchestration, missing values, scaling, predictions
├── datasets/
│   ├── titanic_missing_values.csv # Real missing values for testing Mean/Median/Mode
│   ├── iris.csv                  # Standard 3-class classification
│   └── housing_regression.csv    # Continuous targets for regression
├── app.py                   # One-click launcher script
└── index.html               # Root integrated single-page application
```

---

## 🏃‍♂️ How to Run

### Option 1: Start the Full-Stack Server (FastAPI Backend + Frontend)
In your terminal, navigate to the project directory and run:
```bash
python3 app.py
```
Then open your browser at:
👉 **`http://localhost:8000`**

To view the interactive Swagger API documentation:
👉 **`http://localhost:8000/docs`**

### Option 2: Run Automated Backend Tests
```bash
python3 backend/test_backend.py
```

### Option 3: Open Directly in Browser
You can also directly double-click or open `index.html` in any modern web browser (`file:///Users/harshit9793/Decition%20tree/index.html`).

### Deploy to Render
This repository includes a `render.yaml` Blueprint for deploying the FastAPI backend and frontend as one web service.

1. Push the project, including `render.yaml`, to your GitHub repository.
2. Sign in to [Render](https://dashboard.render.com/) and choose **New** → **Blueprint**.
3. Connect your GitHub account if prompted, then select `harshitsachan830-dev/Diciation-tree-visualization2`.
4. Review the Blueprint service `decision-tree-studio` and click **Apply**. Render installs the packages from `requirements.txt` and starts the app using the configured command.
5. Wait for the first deploy to finish, then open the `onrender.com` URL shown on the service page. The health check at `/api/health` should report a healthy service.
6. To publish later changes, push them to the connected branch; Render will deploy them automatically.

The configured free service can spin down when idle, so the first request after a quiet period may take longer. Uploaded datasets and trained models are held in memory and are not durable; restarting or redeploying the service clears them.
