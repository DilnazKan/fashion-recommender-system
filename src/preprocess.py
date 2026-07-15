"""Feature engineering for the fashion recommender.

The raw dataset has no free-text description, so we synthesize a per-product
"content profile" string from its categorical attributes for TF-IDF to work on,
and separately min-max scale the numeric attributes (Price, Rating) so they can
be blended into the similarity computation alongside the text features.
"""

import pandas as pd

CONTENT_COLUMNS = ["Product Name", "Brand", "Category", "Color", "Size"]


def build_content_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `content_profile` text column and scaled numeric columns."""
    df = df.copy()

    df["content_profile"] = df[CONTENT_COLUMNS].astype(str).agg(" ".join, axis=1)

    for col in ["Price", "Rating"]:
        min_val, max_val = df[col].min(), df[col].max()
        df[f"{col}_scaled"] = (df[col] - min_val) / (max_val - min_val)

    return df


def preprocess(raw_path: str, processed_path: str) -> pd.DataFrame:
    """Run the full preprocessing pipeline and write the result to disk."""
    from src.data_loader import load_raw_data

    df = load_raw_data(raw_path)
    df = build_content_profile(df)
    df.to_csv(processed_path, index=False)
    return df


if __name__ == "__main__":
    preprocess(
        raw_path="data/raw/fashion_products.csv",
        processed_path="data/processed/cleaned_fashion_products.csv",
    )
