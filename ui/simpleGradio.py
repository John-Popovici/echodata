# app.py
import os
import tempfile
import numpy as np
import pandas as pd
import gradio as gr

from models.tabfms.tabicl_impl import TabICLImpl


def read_csv_to_df(file_obj) -> pd.DataFrame:
    if file_obj is None:
        raise gr.Error("Please upload the CSV.")
    # gradio gives a tempfile-like object with .name
    return pd.read_csv(file_obj.name)


def infer_columns(file):
    df = read_csv_to_df(file)
    cols = list(map(str, df.columns))
    default = cols[-1] if cols else None
    return gr.update(choices=cols, value=default)


def run_training_and_predict(train_file, test_file, train_target_col, test_target_col):
    # Load CSVs
    train_df = read_csv_to_df(train_file)
    test_df = read_csv_to_df(test_file)

    # Train target (default to last col)
    if train_target_col is None or train_target_col not in train_df.columns:
        train_target_col = train_df.columns[-1]

    # Split into numpy
    y_train = train_df[train_target_col].to_numpy()
    X_train = train_df.drop(columns=[train_target_col]).to_numpy()
    X_test = test_df.drop(columns=[test_target_col]).to_numpy() if (test_target_col and test_target_col in test_df.columns) else test_df.to_numpy()

    # Fit + predict
    model = TabICLImpl().fit(X_train, y_train)
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
        train_df.head(5),
        test_df.head(5),
        preds_df.head(10),
        tmp.name,
        info,
        "N/A" if accuracy is None else accuracy,
    )


with gr.Blocks(title="TabICL — CSV Trainer/Tester") as demo:
    gr.Markdown("## TabICL — Train on CSV, Predict on CSV")

    with gr.Row():
        train_csv = gr.File(label="Training CSV", file_types=[".csv"])
        test_csv = gr.File(label="Test CSV", file_types=[".csv"])

    # target selectors
    train_target = gr.Dropdown(
        label="Training target column (defaults to last)",
        choices=[], value=None, interactive=True,
    )
    test_target = gr.Dropdown(
        label="(Optional) Test target column for accuracy",
        choices=[], value=None, interactive=True,
    )

    # populate choices on upload
    train_csv.change(infer_columns, inputs=train_csv, outputs=train_target)
    test_csv.change(infer_columns, inputs=test_csv, outputs=test_target)

    run_btn = gr.Button("Fit & Predict")

    with gr.Row():
        train_preview = gr.Dataframe(label="Training CSV (head)", interactive=False)
        test_preview = gr.Dataframe(label="Test CSV (head)", interactive=False)
    preds_preview = gr.Dataframe(label="Predictions (first 10)", interactive=False)
    preds_file = gr.File(label="Download predictions.csv")
    info_box = gr.Textbox(label="Info", interactive=False)
    accuracy_box = gr.Number(label="Accuracy (if test has target)", precision=4)

    run_btn.click(
        fn=run_training_and_predict,
        inputs=[train_csv, test_csv, train_target, test_target],
        outputs=[train_preview, test_preview, preds_preview, preds_file, info_box, accuracy_box],
    )

if __name__ == "__main__":
    demo.launch()
