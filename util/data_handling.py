# util/data_handling.py

import pandas as pd
import gradio as gr


def read_csv_to_df(file_obj) -> pd.DataFrame:
    """Reads a csv and returns a a df."""
    if file_obj is None:
        raise gr.Error("Please upload the CSV.")

    # Gradio gives a tempfile-like object with .name
    return pd.read_csv(file_obj.name)

