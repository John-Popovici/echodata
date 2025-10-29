# util/trainer.py

import tempfile
import numpy as np
import pandas as pd
import gradio as gr

from models.model_registry import MODEL_REGISTRY


def run_training_and_predict(
    model_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_target_col: str,
    test_target_col: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Train and predict using the model."""
    # Check targets are valid
    if train_target_col not in train_df.columns:
        raise gr.Error("Please select a valid training target column.")
    if test_target_col != "None" and test_target_col not in test_df.columns:
        raise gr.Error("Please select a valid test target column.")

    # Split train into numpy
    y_train: np.ndarray = train_df[train_target_col].to_numpy()
    X_train: np.ndarray = train_df.drop(columns=[train_target_col]).to_numpy()

    # Split test into numpy
    y_test: np.ndarray | None = None
    X_test: np.ndarray = test_df.to_numpy()
    if test_target_col != "None":
        y_test = test_df[test_target_col].to_numpy()
        X_test = test_df.drop(columns=[test_target_col]).to_numpy()

    # Check arrays must be same width
    if X_train.shape[1] + 1 == X_test.shape[1] and test_target_col == "None":
        raise gr.Error("Test target might not be None: Missmatched column numbers by one.")
    if X_train.shape[1] == X_test.shape[1] + 1 and test_target_col != "None":
        raise gr.Error("Test target might be None: Missmatched column numbers by one.")
    if X_train.shape[1] != X_test.shape[1]:
        raise gr.Error("Please ensure training and test have an equal number of columns.")


    # Get model data
    try:
        ModelClass = MODEL_REGISTRY[model_id]["model"]
        model_name = MODEL_REGISTRY[model_id]["model_name"]
        model_type = MODEL_REGISTRY[model_id]["model_type"]
    except KeyError:
        raise gr.Error("Model missing.")

    # Check data compared to model
    print(f"{model_name}_{model_type}")
    if train_df[train_target_col].nunique() > 20 and model_type == "Classification":
        raise gr.Error("Current model type is classification but more than 20 classes detected.")


    # Fit + predict
    model = ModelClass().fit(X_train, y_train)
    preds = model.predict(X_test)

    # Accuracy (only if test has a target column)
    accuracy = None if test_target_col == "None" else float((preds == y_test).mean())

    # Build prediction df
    # Place predictions adjacent to test value or in rightmost column
    pred_insert_index: int = test_df.shape[1]
    pred_col_name: str = "prediction"
    if test_target_col != "None":
        pred_insert_index = test_df.columns.get_loc(test_target_col) + 1
        pred_col_name: str = f"pred_{test_target_col}"

    preds_df = test_df.copy()
    preds_df.insert(pred_insert_index, pred_col_name, preds, True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    preds_df.to_csv(tmp.name, index=False)

    # Construct output information
    info = (
        f"Shapes — X_train: {X_train.shape}, y_train: {y_train.shape}, "
        f"X_test: {X_test.shape}\nModel: {model.model_name}"
    )
    metrics = "N/A" if accuracy is None else str(round(accuracy, 4))

    return (
        preds_df.head(10),
        tmp.name,
        info,
        metrics,
    )
