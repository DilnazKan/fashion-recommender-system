"""Load and validate the raw fashion products dataset."""

import pandas as pd

REQUIRED_COLUMNS = [
    "User ID",
    "Product ID",
    "Product Name",
    "Brand",
    "Category",
    "Price",
    "Rating",
    "Color",
    "Size",
]


def load_raw_data(path: str) -> pd.DataFrame:
    """Read the raw fashion products CSV and validate its schema."""
    df = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    return df
