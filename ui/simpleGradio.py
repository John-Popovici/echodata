# app.py

from typing import Any

import gradio as gr

import models.tabfms  # import all models
from models.model_base import BaseModel
from models.model_registry import MODEL_REGISTRY

from util.data_handling import (
    read_csv_to_df,
    preview_data,
    infer_train_columns,
    infer_test_columns,
    analyze_df,
)
from util.trainer import run_training_and_predict


def update_tasks(selected_model_name: str) -> dict[str, Any]:
    """Update possible tasks based on model."""
    model_cls: type[BaseModel] = MODEL_REGISTRY[selected_model_name]
    return gr.update(choices=model_cls.model_possible_tasks, value=None)


def build_interface() -> gr.Blocks:
    """Build the Gradio interface."""

    with gr.Blocks(title="CSV Trainer/Tester") as demo:
        gr.Markdown("## Train on CSV, Predict on CSV")

        # Store pd.dataframe of csv data
        data_train = gr.State(None)
        data_test = gr.State(None)

        # Select model
        first_model_name: str = (
            "TabPFN" if "TabPFN" in MODEL_REGISTRY
            else next(iter(MODEL_REGISTRY))
        )

        with gr.Row():
            model_name = gr.Dropdown(
                label="Select a model",
                choices=list[str](MODEL_REGISTRY.keys()),
                value=first_model_name,
                interactive=True,
            )

            model_task = gr.Dropdown(
                label="Select a model type",
                choices=MODEL_REGISTRY[first_model_name].model_possible_tasks,
                value=MODEL_REGISTRY[first_model_name].model_possible_tasks[0],
                interactive=True,
            )

        model_name.change(fn=update_tasks, inputs=model_name, outputs=model_task)

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
        with gr.Accordion(label="Data Modification", open=True):
            with gr.Row():
                with gr.Column():
                    add_train_header = gr.Checkbox(
                        value=False,
                        label="Add Header Row for Training Data",
                    )
                    # info_box_train = gr.Textbox(
                    #     label="Training Data Information",
                    #     lines=5,
                    #     max_lines=20,
                    #     interactive=False,
                    # )
                with gr.Column():
                    add_test_header = gr.Checkbox(
                        value=False,
                        label="Add Header Row for Testing Data",
                    )
                    target_empty = gr.Checkbox(
                        value=False,
                        label="Ignore Target Column",
                    )
                    # info_box_test = gr.Textbox(
                    #     label="Testing Data Information",
                    #     lines=5,
                    #     max_lines=20,
                    #     interactive=False,
                    # )

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
        train_csv.change(
            fn=read_csv_to_df, inputs=[train_csv, add_train_header], outputs=data_train
        )
        test_csv.change(
            fn=read_csv_to_df, inputs=[test_csv, add_test_header], outputs=data_test
        )
        add_train_header.change(
            fn=read_csv_to_df, inputs=[train_csv, add_train_header], outputs=data_train
        )
        add_test_header.change(
            fn=read_csv_to_df, inputs=[test_csv, add_test_header], outputs=data_test
        )

        # Populate choices on df change
        data_train.change(
            fn=infer_train_columns, inputs=data_train, outputs=train_target
        )
        data_test.change(fn=infer_test_columns, inputs=data_test, outputs=test_target)

        # Populate info on df change
        # data_train.change(fn=analyze_df, inputs=data_train, outputs=info_box_train)
        # data_test.change(fn=analyze_df, inputs=data_test, outputs=info_box_test)

        # Populate preview on df change
        data_train.change(fn=preview_data, inputs=data_train, outputs=train_preview)
        data_test.change(fn=preview_data, inputs=data_test, outputs=test_preview)

        # Fit and Predict
        run_btn = gr.Button("Fit & Predict")

        # Result previews
        preds_preview = gr.Dataframe(label="Predictions Preview", interactive=False)
        preds_file = gr.File(label="Download predictions.csv")
        info_box = gr.Textbox(label="Info", lines=5, max_lines=20, interactive=False)
        scores_box = gr.Dataframe(
            label="Evaluations (if test has target)", interactive=False
        )

        run_btn.click(
            fn=run_training_and_predict,
            inputs=[
                model_name,
                model_task,
                data_train,
                data_test,
                train_target,
                test_target,
                target_empty,
            ],
            outputs=[preds_preview, preds_file, info_box, scores_box],
        )

        return demo


def run_ui() -> None:
    """Launches the Gradio app."""
    demo: gr.Blocks = build_interface()
    demo.launch()
