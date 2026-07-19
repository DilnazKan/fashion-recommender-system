# Fashion Recommender System

A **content-based recommendation engine** for fashion products with a
**feedback-driven learning loop**: recommendations improve as users
like or dislike them. Includes a live **Streamlit demo**.

## How it works

**1. Item profiles.** The dataset (1,000 products: name, brand, category, color,
size, price, rating) has no free-text descriptions, so attributes are
synthesized into a text profile per item
(e.g. `"Dress Adidas Men's Fashion Black XL"`).

**2. Vectorization.** Each profile is turned into a TF-IDF vector; **price and
rating** are scaled to [0, 1] and blended in as numeric features.

**3. Similarity.** Cosine similarity across all item pairs produces a full
similarity matrix.

**4. Recommendation.** For any product, the engine returns the top-N most
similar items — same category/brand/color cluster, close in price and rating.

**5. Feedback loop.** In the app, users can 👍/👎 each recommendation. Every
click updates a persistent boost score for that item pair: liked items rise in
future rankings, disliked items sink. Scores are saved to disk, so learning
survives restarts — a real (not simulated) improvement mechanism.

## Interfaces

- **Jupyter notebooks** — the data-science story: EDA → preprocessing → model build & demo
- **Streamlit app** — live, clickable demo with feedback buttons

```bash
streamlit run app/[имя файла].py
```

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `TF-IDF` · `cosine similarity` · `Streamlit`

## Why content-based?

Works without user interaction history — no cold-start problem for new items.
The feedback loop then layers lightweight personalization on top, moving the
system toward hybrid recommendation as real usage data accumulates.
