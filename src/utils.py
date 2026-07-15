"""Shared helpers for persisting recommender artifacts."""

import pickle
from pathlib import Path
from typing import Any


def save_artifact(obj: Any, path: str) -> None:
    """Pickle an object (vectorizer, similarity matrix, etc.) to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_artifact(path: str) -> Any:
    """Load a previously pickled artifact from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
