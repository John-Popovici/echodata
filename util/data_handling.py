# util/data_handling.py

import pandas as pd
from gradio.utils import NamedString


def read_csv_to_df(file: NamedString | None, add_header: bool) -> pd.DataFrame | None:
    """Reads a CSV and returns a a df."""
    if file is None:
        return None
    if not add_header:
        return pd.read_csv(file)

    # Add header row to file
    df: pd.DataFrame = pd.read_csv(file, header=None)
    df.columns = [f"col{i}" for i in range(df.shape[1])]
    return df

