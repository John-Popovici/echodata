# app.py

import gradio as gr

from util.data_handling import read_csv_to_df, preview_data, infer_train_columns, infer_test_columns, analyze_df
from util.trainer import run_training_and_predict

from models.model_registry import MODEL_REGISTRY


def build_interface() -> gr.Blocks:
    """Build the Gradio interface."""

    with gr.Blocks(title="CSV Trainer/Tester") as demo:
        gr.Markdown("## Train on CSV, Predict on CSV")

        # Store pd.dataframe of csv data
        data_train = gr.State(None)
        data_test = gr.State(None)

        # Select model
        model_names = list(MODEL_REGISTRY.keys())
        model_name = gr.Dropdown(
            label="Select a model",
            choices=model_names,
            value=model_names[0],
            interactive=True,
        )

        # CSV upload
        with gr.Row():
            train_csv = gr.File(label="Training CSV", file_types=[".csv"])
            test_csv = gr.File(label="Test CSV", file_types=[".csv"])

        # CSV target selectors
        with gr.Row():
            train_target = gr.Dropdown(
                label="Training target column",
                choices=[],
                value=None,
                interactive=True,
            )
            test_target = gr.Dropdown(
                label="Test target column",
                choices=[],
                value=None,
                interactive=True,
            )

        # CSV info display
        with gr.Accordion(label="Data Information", open=True):
            with gr.Row():
                add_train_header = gr.Checkbox(
                    value=False,
                    label="Add Header Row for Training Data",
                )
                add_test_header = gr.Checkbox(
                    value=False,
                    label="Add Header Row for Testing Data",
                )

            with gr.Row():
                info_box_train = gr.Textbox(
                    label="Training Data Information",
                    lines=5,
                    max_lines=20,
                    interactive=False,
                )
                info_box_test = gr.Textbox(
                    label="Testing Data Information",
                    lines=5,
                    max_lines=20,
                    interactive=False,
                )

        # CSV preview
        with gr.Row():
            train_preview = gr.Dataframe(
                label="Training Data Preview",
                value=None,
                interactive=False,
            )
            test_preview = gr.Dataframe(
                label="Test Data Preview",
                value=None,
                interactive=False,
            )

        # Save CSV to state
        train_csv.change(fn=read_csv_to_df, inputs=[train_csv, add_train_header], outputs=data_train)
        test_csv.change(fn=read_csv_to_df, inputs=[test_csv, add_test_header], outputs=data_test)
        add_train_header.change(fn=read_csv_to_df, inputs=[train_csv, add_train_header], outputs=data_train)
        add_test_header.change(fn=read_csv_to_df, inputs=[test_csv, add_test_header], outputs=data_test)

        # Populate choices on df change
        data_train.change(fn=infer_train_columns, inputs=data_train, outputs=train_target)
        data_test.change(fn=infer_test_columns, inputs=data_test, outputs=test_target)

        # Populate info on df change
        data_train.change(fn=analyze_df, inputs=data_train, outputs=info_box_train)
        data_test.change(fn=analyze_df, inputs=data_test, outputs=info_box_test)

        # Populate preview on df change
        data_train.change(fn=preview_data, inputs=data_train, outputs=train_preview)
        data_test.change(fn=preview_data, inputs=data_test, outputs=test_preview)

        # Fit and Predict
        run_btn = gr.Button("Fit & Predict")

        # Result previews
        preds_preview = gr.Dataframe(label="Predictions Preview", interactive=False)
        preds_file = gr.File(label="Download predictions.csv")
        info_box = gr.Textbox(label="Info", interactive=False)
        accuracy_box = gr.Textbox(label="Metrics", interactive=False)

        run_btn.click(
            fn=run_training_and_predict,
            inputs=[model_name, data_train, data_test, train_target, test_target],
            outputs=[preds_preview, preds_file, info_box, accuracy_box],
        )

        return demo


def run_ui() -> None:
    """Launches the Gradio app."""
    demo: gr.Blocks = build_interface()
    demo.launch()
