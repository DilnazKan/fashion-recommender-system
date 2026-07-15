"""Durable audit log for user feedback on recommendations.

This is separate from the live `feedback_boost` state carried by
`FashionRecommender` (which actually affects ranking) — this module just
gives an inspectable, append-only record of every like/dislike event.
"""

import os
from datetime import datetime, timezone

import pandas as pd

LOG_COLUMNS = ["timestamp", "query_product_id", "rec_product_id", "signal"]


def log_feedback(query_product_id: int, rec_product_id: int, signal: int, path: str) -> None:
    """Append one feedback event (signal: +1 like, -1 dislike) to the CSV log."""
    row = pd.DataFrame(
        [[datetime.now(timezone.utc).isoformat(), query_product_id, rec_product_id, signal]],
        columns=LOG_COLUMNS,
    )
    write_header = not os.path.exists(path)
    row.to_csv(path, mode="a", header=write_header, index=False)


def load_feedback_log(path: str) -> pd.DataFrame:
    """Read the feedback log, returning an empty frame if it doesn't exist yet."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path)
