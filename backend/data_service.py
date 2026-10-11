"""
Model Mind - Decision Tree Studio Data Service
Handles CSV parsing, column type inference, missing value imputation (Mean, Median, Mode),
and feature scaling (StandardScaler, MinMaxScaler, RobustScaler).
"""

import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

# In-memory storage for uploaded and pre-loaded datasets
DATASETS: Dict[str, pd.DataFrame] = {}
DATASET_METADATA: Dict[str, Dict[str, Any]] = {}

def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        else:
            types[col] = "categorical"
    return types

def get_missing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(sum(missing_counts.values()))
    columns_with_missing = {k: int(v) for k, v in missing_counts.items() if v > 0}
    return {
        "total_missing_cells": total_missing,
        "columns_with_missing": columns_with_missing,
        "has_missing": total_missing > 0
    }

def register_dataset(df: pd.DataFrame, name: str = "dataset") -> Dict[str, Any]:
    dataset_id = str(uuid.uuid4())[:8]
    DATASETS[dataset_id] = df.copy()
    
    col_types = infer_column_types(df)
    missing_sum = get_missing_summary(df)
    
    meta = {
        "dataset_id": dataset_id,
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "inferred_types": col_types,
        "missing_summary": missing_sum,
        "preview": df.head(10).replace({np.nan: None}).to_dict(orient="records")
    }
    DATASET_METADATA[dataset_id] = meta
    return meta

def get_dataset(dataset_id: str) -> Optional[pd.DataFrame]:
    return DATASETS.get(dataset_id)

def preprocess_dataset(
    dataset_id: str,
    impute_strategy: Optional[str] = None, # 'mean', 'median', 'most_frequent', 'drop'
    impute_columns: Optional[List[str]] = None,
    scale_method: Optional[str] = None,    # 'standard', 'minmax', 'robust'
    scale_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    df = DATASETS.get(dataset_id)
    if df is None:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df_processed = df.copy()
    actions_taken = []
    
    # 1. Missing Value Imputation
    if impute_strategy:
        target_cols = impute_columns or list(df_processed.columns)
        if impute_strategy == "drop":
            before_len = len(df_processed)
            df_processed = df_processed.dropna(subset=target_cols)
            actions_taken.append(f"Dropped {before_len - len(df_processed)} rows containing missing values")
        else:
            # SimpleImputer strategies: 'mean', 'median', 'most_frequent' (mode), 'constant'
            strategy = "most_frequent" if impute_strategy in ["mode", "most_frequent"] else impute_strategy
            for col in target_cols:
                if col in df_processed.columns and df_processed[col].isnull().any():
                    col_is_num = pd.api.types.is_numeric_dtype(df_processed[col])
                    actual_strat = strategy if (col_is_num or strategy == "most_frequent") else "most_frequent"
                    imputer = SimpleImputer(strategy=actual_strat)
                    vals = df_processed[[col]].values
                    df_processed[col] = imputer.fit_transform(vals).ravel()
                    actions_taken.append(f"Imputed missing values in '{col}' using {actual_strat}")
    
    # 2. Feature Scaling
    if scale_method and scale_method.lower() != "none":
        scaler_cls = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler
        }.get(scale_method.lower())
        
        if scaler_cls:
            target_scale_cols = [
                c for c in (scale_columns or df_processed.columns)
                if c in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[c])
            ]
            if target_scale_cols:
                # Ensure no NaNs before scaling
                for c in target_scale_cols:
                    if df_processed[c].isnull().any():
                        median_val = df_processed[c].median()
                        df_processed[c] = df_processed[c].fillna(median_val)
                
                scaler = scaler_cls()
                df_processed[target_scale_cols] = scaler.fit_transform(df_processed[target_scale_cols])
                actions_taken.append(f"Applied {scale_method.capitalize()} scaling to: {', '.join(target_scale_cols)}")

    # Update stored dataset
    DATASETS[dataset_id] = df_processed
    new_meta = register_dataset(df_processed, name=DATASET_METADATA.get(dataset_id, {}).get("name", "dataset"))
    new_meta["actions_taken"] = actions_taken
    return new_meta

def load_default_datasets():
    # 1. Iris
    from sklearn.datasets import load_iris, load_wine
    iris = load_iris(as_frame=True)
    iris_df = iris.frame.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "species"
    })
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    iris_df["species"] = iris_df["species"].map(species_map)
    register_dataset(iris_df, name="iris")
    
    # 2. Titanic with realistic missing values
    titanic_data = [
        {"survived": 0, "pclass": 3, "sex": "male", "age": 22.0, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"},
        {"survived": 1, "pclass": 1, "sex": "female", "age": 38.0, "sibsp": 1, "parch": 0, "fare": 71.28, "embarked": "C"},
        {"survived": 1, "pclass": 3, "sex": "female", "age": 26.0, "sibsp": 0, "parch": 0, "fare": 7.92, "embarked": "S"},
        {"survived": 1, "pclass": 1, "sex": "female", "age": 35.0, "sibsp": 1, "parch": 0, "fare": 53.10, "embarked": "S"},
        {"survived": 0, "pclass": 3, "sex": "male", "age": 35.0, "sibsp": 0, "parch": 0, "fare": 8.05, "embarked": "S"},
        {"survived": 0, "pclass": 3, "sex": "male", "age": np.nan, "sibsp": 0, "parch": 0, "fare": 8.46, "embarked": "Q"},
        {"survived": 0, "pclass": 1, "sex": "male", "age": 54.0, "sibsp": 0, "parch": 0, "fare": 51.86, "embarked": "S"},
        {"survived": 0, "pclass": 3, "sex": "male", "age": 2.0, "sibsp": 3, "parch": 1, "fare": 21.08, "embarked": "S"},
        {"survived": 1, "pclass": 3, "sex": "female", "age": 27.0, "sibsp": 0, "parch": 2, "fare": 11.13, "embarked": "S"},
        {"survived": 1, "pclass": 2, "sex": "female", "age": 14.0, "sibsp": 1, "parch": 0, "fare": 30.07, "embarked": "C"},
        {"survived": 1, "pclass": 3, "sex": "female", "age": 4.0, "sibsp": 1, "parch": 1, "fare": 16.70, "embarked": "S"},
        {"survived": 1, "pclass": 1, "sex": "female", "age": 58.0, "sibsp": 0, "parch": 0, "fare": 26.55, "embarked": "S"},
        {"survived": 0, "pclass": 3, "sex": "male", "age": 20.0, "sibsp": 0, "parch": 0, "fare": 8.05, "embarked": "S"},
        {"survived": 0, "pclass": 3, "sex": "male", "age": 39.0, "sibsp": 1, "parch": 5, "fare": 31.28, "embarked": "S"},
        {"survived": 0, "pclass": 3, "sex": "female", "age": 14.0, "sibsp": 0, "parch": 0, "fare": 7.85, "embarked": "S"},
        {"survived": 1, "pclass": 2, "sex": "female", "age": 55.0, "sibsp": 0, "parch": 0, "fare": 16.00, "embarked": "S"},
        {"survived": 0, "pclass": 3, "sex": "male", "age": 2.0, "sibsp": 4, "parch": 1, "fare": 29.12, "embarked": "Q"},
        {"survived": 1, "pclass": 2, "sex": "male", "age": np.nan, "sibsp": 0, "parch": 0, "fare": 13.00, "embarked": "S"},
        {"survived": 0, "pclass": 3, "sex": "female", "age": 31.0, "sibsp": 1, "parch": 0, "fare": 18.00, "embarked": "S"},
        {"survived": 1, "pclass": 3, "sex": "female", "age": np.nan, "sibsp": 0, "parch": 0, "fare": 7.22, "embarked": "C"},
        {"survived": 0, "pclass": 2, "sex": "male", "age": 35.0, "sibsp": 0, "parch": 0, "fare": 26.00, "embarked": "S"},
        {"survived": 1, "pclass": 2, "sex": "male", "age": 34.0, "sibsp": 0, "parch": 0, "fare": 13.00, "embarked": "S"},
        {"survived": 1, "pclass": 3, "sex": "female", "age": 15.0, "sibsp": 0, "parch": 0, "fare": 8.03, "embarked": "Q"},
        {"survived": 1, "pclass": 1, "sex": "male", "age": 28.0, "sibsp": 0, "parch": 0, "fare": 35.50, "embarked": "S"},
        {"survived": 0, "pclass": 3, "sex": "female", "age": 8.0, "sibsp": 3, "parch": 1, "fare": 21.08, "embarked": "S"},
        {"survived": 1, "pclass": 3, "sex": "female", "age": 38.0, "sibsp": 1, "parch": 5, "fare": 31.39, "embarked": "S"},
        {"survived": 0, "pclass": 3, "sex": "male", "age": np.nan, "sibsp": 0, "parch": 0, "fare": np.nan, "embarked": "C"},
        {"survived": 0, "pclass": 1, "sex": "male", "age": 19.0, "sibsp": 3, "parch": 2, "fare": 263.00, "embarked": "S"},
        {"survived": 1, "pclass": 3, "sex": "female", "age": np.nan, "sibsp": 0, "parch": 0, "fare": 7.88, "embarked": "Q"},
        {"survived": 0, "pclass": 3, "sex": "male", "age": np.nan, "sibsp": 0, "parch": 0, "fare": 7.90, "embarked": "S"}
    ]
    titanic_df = pd.DataFrame(titanic_data)
    register_dataset(titanic_df, name="titanic")
    
    # 3. Housing Prices (Regression)
    housing_data = [
        {"crim": 0.00632, "rm": 6.575, "age": 65.2, "dis": 4.0900, "tax": 296.0, "ptratio": 15.3, "lstat": 4.98, "price_k": 240.0},
        {"crim": 0.02731, "rm": 6.421, "age": 78.9, "dis": 4.9671, "tax": 242.0, "ptratio": 17.8, "lstat": 9.14, "price_k": 216.0},
        {"crim": 0.02729, "rm": 7.185, "age": 61.1, "dis": 4.9671, "tax": 242.0, "ptratio": 17.8, "lstat": 4.03, "price_k": 347.0},
        {"crim": 0.03237, "rm": 6.998, "age": 45.8, "dis": 6.0622, "tax": 222.0, "ptratio": 18.7, "lstat": 2.94, "price_k": 334.0},
        {"crim": 0.06905, "rm": 7.147, "age": 54.2, "dis": 6.0622, "tax": 222.0, "ptratio": 18.7, "lstat": 5.33, "price_k": 362.0},
        {"crim": 0.02985, "rm": 6.430, "age": 58.7, "dis": 6.0622, "tax": 222.0, "ptratio": 18.7, "lstat": 5.21, "price_k": 287.0},
        {"crim": 0.08829, "rm": 6.012, "age": 66.6, "dis": 5.5605, "tax": 311.0, "ptratio": 15.2, "lstat": 12.43, "price_k": 229.0},
        {"crim": 0.14455, "rm": 6.172, "age": 96.1, "dis": 2.7478, "tax": 666.0, "ptratio": 20.2, "lstat": 19.15, "price_k": 271.0},
        {"crim": 0.21124, "rm": 5.631, "age": 100.0, "dis": 1.4254, "tax": 666.0, "ptratio": 20.2, "lstat": 29.93, "price_k": 165.0},
        {"crim": 0.17004, "rm": 6.004, "age": 85.9, "dis": 2.0720, "tax": 666.0, "ptratio": 20.2, "lstat": 17.10, "price_k": 189.0}
    ]
    housing_df = pd.DataFrame(housing_data)
    register_dataset(housing_df, name="housing")
