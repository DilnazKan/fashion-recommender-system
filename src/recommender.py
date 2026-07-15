"""Content-based fashion recommender.

Vectorizes each product's synthesized attribute profile with TF-IDF, blends in
scaled numeric features (Price, Rating), and ranks candidates by cosine
similarity to produce "similar item" recommendations.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.feedback import log_feedback
from src.utils import load_artifact, save_artifact

NUMERIC_COLUMNS = ["Price_scaled", "Rating_scaled"]
DISPLAY_COLUMNS = ["Product ID", "Product Name", "Brand", "Category", "Color", "Price", "Rating"]

FEEDBACK_LEARNING_RATE = 0.05
FEEDBACK_BOOST_CLIP = 0.3
FEEDBACK_LOG_PATH = "outputs/feedback_log.csv"


class FashionRecommender:
    """Content-based recommender using TF-IDF + cosine similarity.

    Recommendations improve with usage: `record_feedback` nudges a persistent
    item-item boost that is added to the base cosine similarity score on every
    subsequent `recommend` call, so liked/disliked pairs shift in rank over time.
    """

    def __init__(self, numeric_weight: float = 0.5):
        self.numeric_weight = numeric_weight
        self.vectorizer = TfidfVectorizer()
        self.similarity_matrix = None
        self.df = None
        self.product_id_to_index = None
        self.feedback_boost = {}

    def fit(self, df: pd.DataFrame) -> "FashionRecommender":
        self.df = df.reset_index(drop=True)

        tfidf_matrix = self.vectorizer.fit_transform(self.df["content_profile"])
        numeric_matrix = self.df[NUMERIC_COLUMNS].to_numpy(dtype=float) * self.numeric_weight
        combined_matrix = hstack([tfidf_matrix, numeric_matrix]).tocsr()

        self.similarity_matrix = cosine_similarity(combined_matrix)
        self.product_id_to_index = {
            pid: idx for idx, pid in enumerate(self.df["Product ID"])
        }
        return self

    def recommend(self, product_id: int, top_n: int = 5) -> pd.DataFrame:
        if product_id not in self.product_id_to_index:
            raise ValueError(f"Unknown Product ID: {product_id}")

        idx = self.product_id_to_index[product_id]
        base_scores = self.similarity_matrix[idx]

        boosted_scores = base_scores.copy()
        for (query_id, other_id), boost in self.feedback_boost.items():
            if query_id == product_id and other_id in self.product_id_to_index:
                boosted_scores[self.product_id_to_index[other_id]] += boost

        ranked_indices = np.argsort(boosted_scores)[::-1]
        ranked_indices = [i for i in ranked_indices if i != idx][:top_n]

        result = self.df.loc[ranked_indices, DISPLAY_COLUMNS].copy()
        result["similarity_score"] = boosted_scores[ranked_indices]
        return result.reset_index(drop=True)

    def record_feedback(
        self,
        query_product_id: int,
        rec_product_id: int,
        signal: int,
        log_path: str = FEEDBACK_LOG_PATH,
    ) -> float:
        """Nudge the persistent boost for a (query, recommended) pair.

        `signal` is +1 for like, -1 for dislike. Applied symmetrically since
        "these two go well together" is a mutual signal. Returns the new
        boost value for the pair.
        """
        for key in [(query_product_id, rec_product_id), (rec_product_id, query_product_id)]:
            new_boost = self.feedback_boost.get(key, 0.0) + FEEDBACK_LEARNING_RATE * signal
            new_boost = max(-FEEDBACK_BOOST_CLIP, min(FEEDBACK_BOOST_CLIP, new_boost))
            self.feedback_boost[key] = new_boost

        log_feedback(query_product_id, rec_product_id, signal, log_path)
        return self.feedback_boost[(query_product_id, rec_product_id)]

    def save(self, path: str) -> None:
        save_artifact(
            {
                "vectorizer": self.vectorizer,
                "similarity_matrix": self.similarity_matrix,
                "df": self.df,
                "product_id_to_index": self.product_id_to_index,
                "numeric_weight": self.numeric_weight,
                "feedback_boost": self.feedback_boost,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FashionRecommender":
        state = load_artifact(path)
        instance = cls(numeric_weight=state["numeric_weight"])
        instance.vectorizer = state["vectorizer"]
        instance.similarity_matrix = state["similarity_matrix"]
        instance.df = state["df"]
        instance.product_id_to_index = state["product_id_to_index"]
        instance.feedback_boost = state.get("feedback_boost", {})
        return instance


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_fashion_products.csv")
    model = FashionRecommender().fit(df)
    model.save("outputs/recommender_artifacts.pkl")

    sample_id = int(df["Product ID"].iloc[0])
    print(f"Recommendations for Product ID {sample_id}:")
    print(model.recommend(sample_id, top_n=5))
