"""Streamlit demo for the content-based fashion recommender."""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feedback import load_feedback_log
from src.recommender import FEEDBACK_LOG_PATH, FashionRecommender

ARTIFACTS_PATH = "outputs/recommender_artifacts.pkl"
PROCESSED_DATA_PATH = "data/processed/cleaned_fashion_products.csv"


@st.cache_resource
def load_model() -> FashionRecommender:
    if os.path.exists(ARTIFACTS_PATH):
        return FashionRecommender.load(ARTIFACTS_PATH)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    model = FashionRecommender().fit(df)
    model.save(ARTIFACTS_PATH)
    return model


st.set_page_config(page_title="Fashion Recommender", page_icon="👗", layout="centered")
st.title("👗 Fashion Recommender")
st.caption("Content-based recommendations using TF-IDF + cosine similarity over product attributes.")

model = load_model()
df = model.df

df["label"] = (
    df["Product Name"] + " — " + df["Brand"] + " — " + df["Color"]
    + " (ID " + df["Product ID"].astype(str) + ")"
)

selected_label = st.selectbox("Pick a product", df["label"].sort_values())
selected_row = df[df["label"] == selected_label].iloc[0]
selected_product_id = int(selected_row["Product ID"])

top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

st.subheader("Selected product")
st.table(
    selected_row[["Product ID", "Product Name", "Brand", "Category", "Color", "Size", "Price", "Rating"]]
    .to_frame()
    .T
)

if st.button("Get recommendations", type="primary"):
    st.session_state["query_product_id"] = selected_product_id
    st.session_state["top_n"] = top_n

if "query_product_id" in st.session_state:
    query_product_id = st.session_state["query_product_id"]
    recommendations = model.recommend(query_product_id, top_n=st.session_state["top_n"])

    st.subheader(f"Top {st.session_state['top_n']} similar products")
    st.caption("Rate a recommendation 👍/👎 — the engine remembers it and re-ranks future results for this product.")

    for _, rec in recommendations.iterrows():
        rec_product_id = int(rec["Product ID"])
        cols = st.columns([4, 1, 1])
        cols[0].write(
            f"**{rec['Product Name']}** — {rec['Brand']} — {rec['Color']} "
            f"— ${rec['Price']} — ⭐ {rec['Rating']:.2f} — score `{rec['similarity_score']:.3f}` "
            f"(ID {rec_product_id})"
        )
        if cols[1].button("👍", key=f"like_{rec_product_id}"):
            model.record_feedback(query_product_id, rec_product_id, signal=+1, log_path=FEEDBACK_LOG_PATH)
            model.save(ARTIFACTS_PATH)
            st.rerun()
        if cols[2].button("👎", key=f"dislike_{rec_product_id}"):
            model.record_feedback(query_product_id, rec_product_id, signal=-1, log_path=FEEDBACK_LOG_PATH)
            model.save(ARTIFACTS_PATH)
            st.rerun()

    st.bar_chart(recommendations.set_index("Product Name")["similarity_score"])

st.divider()
feedback_log = load_feedback_log(FEEDBACK_LOG_PATH)
st.caption(f"📈 Learning progress: {len(feedback_log)} feedback events recorded so far.")
