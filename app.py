"""
Outfit Recommender System
=========================
Adapted from Practical 8 (Parts I, II, III) - movie recommender -> outfit recommender.

The structure below follows the practicals directly:
  Section 1 : Data Preparation            (Practical 8 Part II, Section 1)
  Section 2 : Features Generation         (Practical 8 Part II, Section 2)
  Section 3 : Content-Based Recommender   (Practical 8 Part II, Section 3)
  Section 4 : Collaborative Filtering     (Practical 8 Part III)
  Section 5 : Hybrid (content + collaborative)

Main change from the movie version: a movie recommender returns items in the
SAME category (Spider-Man -> more superhero films). An outfit recommender must
return a DIFFERENT category (a shirt -> trousers/skirts), so we filter the
candidates to complementary categories before ranking them.

Data required (real Kaggle photos):
  sample_data/styles.csv
  sample_data/images/<id>.jpg
Exported from the Kaggle "Fashion Product Images (Small)" dataset.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

st.set_page_config(page_title="Outfit Recommender", layout="wide")

DATA_DIR = "sample_data"
CSV_PATH = os.path.join(DATA_DIR, "styles.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")

# ---------------------------------------------------------------------------
# Clothing category rules (the outfit-specific part of this project)
# ---------------------------------------------------------------------------
TOPS = ["Shirts", "Tshirts", "Tops", "Sweaters", "Sweatshirts", "Kurtas"]
BOTTOMS = ["Jeans", "Trousers", "Skirts", "Shorts", "Track Pants", "Leggings"]
DRESSES = ["Dresses"]
SHOES = ["Casual Shoes", "Heels", "Sandals", "Sports Shoes", "Flats", "Formal Shoes"]

# Which categories COMPLETE a given category
COMPLEMENT = {}
for t in TOPS:
    COMPLEMENT[t] = BOTTOMS + SHOES
for b in BOTTOMS:
    COMPLEMENT[b] = TOPS + SHOES
for d in DRESSES:
    COMPLEMENT[d] = SHOES
for s in SHOES:
    COMPLEMENT[s] = TOPS + BOTTOMS + DRESSES

# Simple colour-matching rule: neutrals go with anything
NEUTRALS = {"Black", "White", "Grey", "Navy Blue", "Beige", "Brown", "Cream"}


def colour_score(c1, c2):
    """1.0 = matches well, 0.8 = same colour, 0.3 = weak match."""
    if c1 in NEUTRALS or c2 in NEUTRALS:
        return 1.0
    if c1 == c2:
        return 0.8
    return 0.3


# ---------------------------------------------------------------------------
# Section 1 - Data Preparation      (Practical 8 Part II, Section 1)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load styles.csv, keep only items we have rules and images for."""
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    df = df[df["articleType"].isin(COMPLEMENT.keys())]
    df = df.dropna(subset=["gender", "articleType", "baseColour", "season", "usage"])
    df = df[df["id"].apply(lambda i: os.path.exists(os.path.join(IMG_DIR, f"{i}.jpg")))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 2 - Features Generation   (Practical 8 Part II, Section 2)
# The movie version used the 'overview' text. Here we build the same kind of
# text from the item's attributes, then TF-IDF + cosine similarity.
# ---------------------------------------------------------------------------
@st.cache_data
def build_similarity(df):
    soup = (
        df["gender"] + " " + df["usage"] + " " + df["season"] + " "
        + df["baseColour"].str.replace(" ", "") + " " + df["articleType"]
    )
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(soup)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)   # same as practical
    return cosine_sim


# ---------------------------------------------------------------------------
# Section 3 - Content-Based Recommender   (Practical 8 Part II, Section 3)
# Same steps as the practical: get index -> score all items -> sort -> top K.
# Added step: keep only COMPLEMENTARY categories, and add a colour score.
# ---------------------------------------------------------------------------
def find_candidates(df, item):
    """Candidate items = complementary categories only.

    We prefer a strict match (same occasion and gender), but if the product
    sample is small that can leave nothing to recommend, so we relax the
    filters step by step instead of returning an empty result.
    """
    base = df[df["articleType"].isin(COMPLEMENT[item["articleType"]])]

    strict = base[
        (base["usage"] == item["usage"])
        & (base["gender"].isin([item["gender"], "Unisex"]))
    ]
    if len(strict) >= 3:
        return strict.copy()

    same_usage = base[base["usage"] == item["usage"]]
    if len(same_usage) >= 3:
        return same_usage.copy()

    return base.copy()


def get_recommendations(item_id, df, cosine_sim, indices, k=5):
    idx = indices[item_id]
    item = df.iloc[idx]

    candidates = find_candidates(df, item)
    if candidates.empty:
        return []

    positions = indices[candidates["id"]].values
    style = cosine_sim[idx, positions]
    colour = candidates["baseColour"].apply(lambda c: colour_score(item["baseColour"], c)).values

    candidates["score"] = 0.5 * style + 0.5 * colour
    candidates = candidates.sort_values("score", ascending=False)
    return balanced_top_k(candidates, k)


def balanced_top_k(candidates, k=5):
    """Take the best item from each clothing group first, then fill the rest.
    Without this, one group (e.g. shoes) can fill all K slots."""
    groups = [TOPS, BOTTOMS, DRESSES, SHOES]
    chosen = []
    for types in groups:
        sub = candidates[candidates["articleType"].isin(types)]
        if not sub.empty:
            chosen.append(sub.iloc[0]["id"])
    for item_id in candidates["id"]:
        if len(chosen) >= k:
            break
        if item_id not in chosen:
            chosen.append(item_id)
    return chosen[:k]


# ---------------------------------------------------------------------------
# Section 4 - Collaborative Filtering   (Practical 8 Part III)
# The practical read real user ratings. The Kaggle catalogue has no purchase
# history, so we simulate users who buy matching pairs together.
# (State this clearly as a limitation in the report.)
# ---------------------------------------------------------------------------
@st.cache_data
def build_collaborative(df, n_users=120):
    rng = np.random.default_rng(42)
    anchors = df[df["articleType"].isin(TOPS + DRESSES)]
    if anchors.empty:
        return pd.DataFrame()

    records = []
    for user in range(n_users):
        for _ in range(10):
            a = anchors.iloc[rng.integers(len(anchors))]
            pool = df[
                df["articleType"].isin(COMPLEMENT[a["articleType"]])
                & (df["usage"] == a["usage"])
            ]
            if pool.empty:
                continue
            b = pool.iloc[rng.integers(len(pool))]
            if colour_score(a["baseColour"], b["baseColour"]) >= 0.8:
                records.append((user, a["id"]))
                records.append((user, b["id"]))

    if not records:
        return pd.DataFrame()

    interactions = pd.DataFrame(records, columns=["user_id", "item_id"]).drop_duplicates()
    interactions["bought"] = 1

    # user-item matrix, exactly like the practical's pivot_table
    user_item = interactions.pivot_table(index="user_id", columns="item_id", values="bought").fillna(0)

    # The practical used corrwith(), which returns NaN on sparse data.
    # cosine_similarity on the same matrix is the same idea without the NaN.
    item_sim = cosine_similarity(user_item.T)
    return pd.DataFrame(item_sim, index=user_item.columns, columns=user_item.columns)


def collaborative_recommendations(item_id, df, item_sim, k=5):
    if item_sim.empty or item_id not in item_sim.index:
        return []

    scores = item_sim[item_id].drop(index=item_id, errors="ignore")
    scores = scores[scores > 0]
    if scores.empty:
        return []

    item = df[df["id"] == item_id].iloc[0]
    candidates = df[
        df["id"].isin(scores.index)
        & df["articleType"].isin(COMPLEMENT[item["articleType"]])
    ].copy()
    if candidates.empty:
        return []

    candidates["score"] = candidates["id"].map(scores)
    candidates = candidates.sort_values("score", ascending=False)
    return balanced_top_k(candidates, k)


# ---------------------------------------------------------------------------
# Section 5 - Hybrid: average the content score and the collaborative score
# ---------------------------------------------------------------------------
def hybrid_recommendations(item_id, df, cosine_sim, indices, item_sim, k=5):
    idx = indices[item_id]
    item = df.iloc[idx]

    candidates = find_candidates(df, item)
    if candidates.empty:
        return []

    positions = indices[candidates["id"]].values
    style = cosine_sim[idx, positions]
    colour = candidates["baseColour"].apply(lambda c: colour_score(item["baseColour"], c)).values
    candidates["content"] = 0.5 * style + 0.5 * colour

    if not item_sim.empty and item_id in item_sim.index:
        candidates["collab"] = candidates["id"].map(item_sim[item_id]).fillna(0)
    else:
        candidates["collab"] = 0.0

    for col in ["content", "collab"]:
        spread = candidates[col].max() - candidates[col].min()
        candidates[col] = (candidates[col] - candidates[col].min()) / spread if spread > 0 else 0

    candidates["score"] = 0.5 * candidates["content"] + 0.5 * candidates["collab"]
    candidates = candidates.sort_values("score", ascending=False)
    return balanced_top_k(candidates, k)


# ---------------------------------------------------------------------------
# Helper: detect the main colour of an uploaded photo
# ---------------------------------------------------------------------------
COLOUR_RGB = {
    "Black": (20, 20, 20), "White": (245, 245, 245), "Grey": (150, 150, 150),
    "Navy Blue": (30, 40, 80), "Blue": (50, 100, 190), "Red": (190, 40, 40),
    "Maroon": (110, 30, 45), "Green": (50, 140, 80), "Yellow": (225, 200, 60),
    "Beige": (220, 205, 175), "Brown": (110, 70, 40), "Pink": (230, 140, 170),
    "Purple": (110, 60, 140), "Orange": (220, 120, 40), "Silver": (192, 192, 192),
}


def detect_colour(image, available):
    pixels = np.array(image.convert("RGB").resize((50, 50))).reshape(-1, 3).mean(axis=0)
    options = {c: rgb for c, rgb in COLOUR_RGB.items() if c in available}
    if not options:
        return available[0]
    return min(options, key=lambda c: np.linalg.norm(np.array(options[c]) - pixels))


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
st.title("Outfit Recommender")

if not os.path.exists(CSV_PATH) or not os.path.isdir(IMG_DIR):
    st.error(
        "Product data not found.\n\n"
        "This app needs the real Kaggle product photos. Add a folder named "
        "`sample_data` to the repository containing `styles.csv` and an "
        "`images` folder, then redeploy."
    )
    st.stop()

df = load_data()
if df.empty:
    st.error("styles.csv loaded, but no rows had a matching image in sample_data/images.")
    st.stop()

cosine_sim = build_similarity(df)
indices = pd.Series(df.index, index=df["id"])     # reverse map, like the practical
item_sim = build_collaborative(df)

st.caption(f"{len(df)} products loaded")

left, right = st.columns([1, 2])

with left:
    st.subheader("1. Upload an item")
    uploaded = st.file_uploader("Photo of a top, bottom, dress or shoe", type=["jpg", "jpeg", "png"])

    detected = None
    if uploaded:
        photo = Image.open(uploaded)
        st.image(photo, caption="Your upload", width='stretch')
        detected = detect_colour(photo, sorted(df["baseColour"].unique()))
        st.success(f"Detected colour: **{detected}**")

    st.subheader("2. Confirm the details")
    group = st.selectbox("Item type", ["Top", "Bottom", "Dress", "Shoes"])
    options = {"Top": TOPS, "Bottom": BOTTOMS, "Dress": DRESSES, "Shoes": SHOES}[group]
    options = [t for t in options if t in set(df["articleType"])] or options
    article_type = st.selectbox("Specific type", options)

    colours = sorted(df["baseColour"].unique())
    colour_index = colours.index(detected) if detected in colours else 0
    colour = st.selectbox("Colour", colours, index=colour_index)

    gender = st.selectbox("Gender", sorted(df["gender"].unique()))
    usage = st.selectbox("Style / occasion", sorted(df["usage"].unique()))
    season = st.selectbox("Season", sorted(df["season"].unique()))
    method = st.radio("Method", ["Hybrid", "Content-based", "Collaborative"])
    search = st.button("Find matching items", type="primary")

with right:
    st.subheader("Recommended items")

    if not search:
        st.info("Fill in the details on the left, then click the button.")
    else:
        # A newly uploaded photo is not in the catalogue, so we match it to the
        # closest existing product and recommend from there.
        pool = df[df["articleType"] == article_type]
        matching_gender = pool[pool["gender"].isin([gender, "Unisex"])]
        if not matching_gender.empty:
            pool = matching_gender

        if pool.empty:
            st.warning("No products of that type in the sample. Try another type.")
        else:
            pool = pool.copy()
            pool["match"] = (
                (pool["usage"] == usage).astype(float)
                + (pool["season"] == season).astype(float)
                + pool["baseColour"].apply(lambda c: colour_score(colour, c))
            )
            anchor_id = pool.sort_values("match", ascending=False).iloc[0]["id"]

            if method == "Content-based":
                results = get_recommendations(anchor_id, df, cosine_sim, indices)
            elif method == "Collaborative":
                results = collaborative_recommendations(anchor_id, df, item_sim)
            else:
                results = hybrid_recommendations(anchor_id, df, cosine_sim, indices, item_sim)

            if not results:
                st.warning("No matches found with this method. Try Hybrid, or change the filters.")
            else:
                names = df.set_index("id")["productDisplayName"]
                columns = st.columns(len(results))
                for column, result_id in zip(columns, results):
                    path = os.path.join(IMG_DIR, f"{result_id}.jpg")
                    if os.path.exists(path):
                        column.image(path, width='stretch')
                    column.caption(names.get(result_id, str(result_id)))
                st.caption(f"Method: **{method}**")
