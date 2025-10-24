# app.py

import gradio as gr

from util.data_handling import read_csv_to_df
from util.trainer import run_training_and_predict

from models.model_registry import MODEL_REGISTRY


def infer_columns(file) -> gr.Dropdown:
    """Infer column labels from csv."""
    if file is None:
        return gr.Dropdown(choices=[], value=None)

    df = read_csv_to_df(file)
    cols = list(map(str, df.columns))
    default_label = cols[-1] if cols else None
    return gr.Dropdown(choices=cols, value=default_label)


def preview_data(file) -> gr.Dataframe:
    """Preview 5 rows from csv."""
    if file is None:
        return gr.Dataframe(value=None)

    df = read_csv_to_df(file)
    return gr.Dataframe(value=df.head(5))


def gather_data(file) -> str:
    """Gather and present meta-data about csv."""
    if file is None:
        return gr.Textbox(value=None)

    df = read_csv_to_df(file)

    shape_info = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
    column_info = "\n".join([f"> {col}: {df[col].dtype}" for col in df.columns])

    info = f"{shape_info}\nColumn info:\n{column_info}"
    return info


def build_interface() -> gr.Blocks:
    """Build the Gradio interface."""

    with gr.Blocks(title="CSV Trainer/Tester") as demo:
        gr.Markdown("## Train on CSV, Predict on CSV")

        model_names = list(MODEL_REGISTRY.keys())

        model_name = gr.Dropdown(
            label="Select a model",
            choices=model_names,
            value=model_names[0],
            interactive=True,
        )

        with gr.Row():
            train_csv = gr.File(label="Training CSV", file_types=[".csv"])
            test_csv = gr.File(label="Test CSV", file_types=[".csv"])

        # Target selectors
        with gr.Row():
            train_target = gr.Dropdown(
                label="Training target column (defaults to last)",
                choices=[],
                value=None,
                interactive=True,
            )
            test_target = gr.Dropdown(
                label="(Optional) Test target column for accuracy",
                choices=[],
                value=None,
                interactive=True,
            )

        # Populate choices on upload
        train_csv.change(fn=infer_columns, inputs=train_csv, outputs=train_target)
        test_csv.change(fn=infer_columns, inputs=test_csv, outputs=test_target)

        # CSV info
        with gr.Row():
            info_box_train = gr.Textbox(
                label="Training Data Info",
                lines=5,
                interactive=False,
            )
            info_box_test = gr.Textbox(
                label="Testing Data Info",
                lines=5,
                interactive=False,
            )

        # Populate info on upload
        train_csv.change(fn=gather_data, inputs=train_csv, outputs=info_box_train)
        test_csv.change(fn=gather_data, inputs=test_csv, outputs=info_box_test)

        with gr.Row():
            train_preview = gr.Dataframe(
                label="Training CSV (head)",
                value=None,
                interactive=False,
            )
            test_preview = gr.Dataframe(
                label="Test CSV (head)",
                value=None,
                interactive=False,
            )

        # Populate preview on upload
        train_csv.change(fn=preview_data, inputs=train_csv, outputs=train_preview)
        test_csv.change(fn=preview_data, inputs=test_csv, outputs=test_preview)

        run_btn = gr.Button("Fit & Predict")

        preds_preview = gr.Dataframe(label="Predictions (first 10)", interactive=False)
        preds_file = gr.File(label="Download predictions.csv")
        info_box = gr.Textbox(label="Info", interactive=False)
        accuracy_box = gr.Number(label="Accuracy (if test has target)", precision=4)

        run_btn.click(
            fn=run_training_and_predict,
            inputs=[
                model_name,
                train_csv,
                test_csv,
                train_target,
                test_target,
            ],
            outputs=[
                preds_preview,
                preds_file,
                info_box,
                accuracy_box,
            ],
        )

        return demo


def run_ui():
    """Launches the Gradio app."""
    demo = build_interface()
    demo.launch()
