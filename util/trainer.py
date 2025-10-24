# util/trainer.py

import tempfile
import numpy as np
import pandas as pd
import gradio as gr

from util.data_handling import read_csv_to_df

from models.model_registry import MODEL_REGISTRY


def run_training_and_predict(model_name, train_file, test_file, train_target_col, test_target_col):
    """Train and predict using the model."""
    # Load CSVs
    train_df = read_csv_to_df(train_file)
    test_df = read_csv_to_df(test_file)

    # Train target (default to last col)
    if (train_target_col is None) or (train_target_col not in train_df.columns):
        train_target_col = train_df.columns[-1]

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
        "N/A" if accuracy is None else accuracy,
    )
