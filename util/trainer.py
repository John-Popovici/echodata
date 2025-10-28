# util/trainer.py

import tempfile
import numpy as np
import pandas as pd
import gradio as gr

from models.model_registry import MODEL_REGISTRY


def run_training_and_predict(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_target_col: str,
    test_target_col: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Train and predict using the model."""
    # Train target (default to last col)
    if (train_target_col is None) or (train_target_col not in train_df.columns):
        train_target_col = str(train_df.columns[-1])

    # Split into numpy
    y_train: np.ndarray = train_df[train_target_col].to_numpy()
    X_train: np.ndarray = train_df.drop(columns=[train_target_col]).to_numpy()

    X_test = (
        test_df.drop(columns=[test_target_col]).to_numpy()
        if (test_target_col and test_target_col in test_df.columns)
        else test_df.to_numpy()
    )

    # Fit + predict
    try:
        ModelClass = MODEL_REGISTRY[model_name]
    except KeyError:
        gr.Error("Please select a valid model.")

    model = ModelClass().fit(X_train, y_train)
    preds = model.predict(X_test)

    # Accuracy (only if test has a target column)
    accuracy = None
    if test_target_col and test_target_col in test_df.columns:
        y_test = test_df[test_target_col].to_numpy()
        accuracy = float((preds == y_test).mean())

    # Build outputs
    preds_df = pd.DataFrame({"prediction": preds})
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    preds_df.to_csv(tmp.name, index=False)

    info = (
        f"Shapes — X_train: {X_train.shape}, y_train: {y_train.shape}, "
        f"X_test: {X_test.shape}\nModel: {model.model_name}"
    )
    return (
        preds_df.head(10),
        tmp.name,
        info,
        "N/A" if accuracy is None else str(accuracy),
    )
