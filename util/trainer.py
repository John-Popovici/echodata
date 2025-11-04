# util/trainer.py

from models.tabfms.tabularfm_protocol import TabularFMProtocol


import tempfile
import numpy as np
import pandas as pd
import gradio as gr

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    root_mean_squared_error,
)

from models.model_registry import MODEL_REGISTRY


def evaluation_scores(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Generate prediction scores."""
    scores = dict[str, float]()
    if task == "Classification":
        scores["Accuracy"] = float(accuracy_score(y_true, y_pred))
        scores["Precision (Macro)"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        scores["Recall (Macro)"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        scores["F1 (Macro)"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        scores["Precision (Micro)"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        scores["Recall (Micro)"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        scores["F1 (Micro)"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    else:
        scores["MAE"] = float(mean_absolute_error(y_true, y_pred))
        scores["MSE"] = float(mean_squared_error(y_true, y_pred))
        scores["RMSE"] = float(root_mean_squared_error(y_true, y_pred))
        scores["R2"] = float(r2_score(y_true, y_pred))
        scores["Explained Variance"] = float(explained_variance_score(y_true, y_pred))
    return scores


def load_and_validate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_target_col: str,
    test_target_col: str,
    target_empty: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load and validate dataframes."""
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
    # If target present, remove from features
    if test_target_col != "None":
        y_test = test_df[test_target_col].to_numpy()
        X_test = test_df.drop(columns=[test_target_col]).to_numpy()
    # If target empty, discard it
    if target_empty:
        y_test = None

    # Check arrays must be same width
    if X_train.shape[1] + 1 == X_test.shape[1] and test_target_col == "None":
        raise gr.Error("Test target might not be None: Missmatched column numbers by one.")
    if X_train.shape[1] == X_test.shape[1] + 1 and test_target_col != "None":
        raise gr.Error("Test target might be None: Missmatched column numbers by one.")
    if X_train.shape[1] != X_test.shape[1]:
        raise gr.Error("Please ensure training and test have an equal number of columns.")

    return X_train, y_train, X_test, y_test


def run_training_and_predict(
    model_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_target_col: str,
    test_target_col: str,
    target_empty: bool,
) -> tuple[pd.DataFrame, str, str, str]:
    """Train and predict using the model."""
    # Load and validate data
    X_train, y_train, X_test, y_test = load_and_validate(train_df, test_df, train_target_col, test_target_col, target_empty)

    # Get model data
    try:
        ModelClass: type[TabularFMProtocol] = MODEL_REGISTRY[model_id]["model"]
        model_name: str = MODEL_REGISTRY[model_id]["model_name"]
        model_type: str = MODEL_REGISTRY[model_id]["model_type"]
    except KeyError:
        raise gr.Error("Model missing.")


    # Fit + predict
    model = ModelClass().fit(X_train, y_train)
    y_pred = model.predict(X_test)


    # Build prediction df
    preds_df = test_df.copy()
    # Place predictions adjacent to test value or overwrite or in rightmost column
    # Name predictions column based on target column
    pred_insert_index: int = test_df.shape[1]
    pred_col_name: str = "prediction"
    if test_target_col != "None":
        pred_col_name: str = f"pred_{test_target_col}"
        if target_empty:
            pred_insert_index = test_df.columns.get_loc(test_target_col)
            preds_df.drop(test_target_col, axis=1, inplace=True)
        else:
            pred_insert_index = test_df.columns.get_loc(test_target_col) + 1
    preds_df.insert(pred_insert_index, pred_col_name, y_pred, allow_duplicates=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    preds_df.to_csv(tmp.name, index=False)

    # Construct output information
    info = (
        f"Shapes:\n"
        f"> X_train: {X_train.shape}\n> y_train: {y_train.shape}\n"
        f"> X_test: {X_test.shape}\n> y_test: {y_test.shape if y_test is not None else 'N/A'}\n"
        f"Model: {model.model_name}\n"
        f"> model_id: {model_id}\n"
        f"> model_name: {model_name}\n"
        f"> model_type: {model_type}\n"
    )
    scores: pd.DataFrame | None = None
    if y_test is not None:
        try:
            scores = pd.DataFrame([evaluation_scores(model_type, y_test, y_pred.astype(y_test.dtype))])
        except Exception as e:
            # raise gr.Error("Failed to generate scores.")
            print(e)
            pass

    return (
        preds_df.head(10),
        tmp.name,
        info,
        scores,
    )

