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

# Which genders may be mixed. A women's top may be paired with women's or
# unisex items, but never with menswear.
GENDER_GROUPS = {
    "Men": ["Men", "Unisex", "Boys"],
    "Boys": ["Boys", "Men", "Unisex"],
    "Women": ["Women", "Unisex", "Girls"],
    "Girls": ["Girls", "Women", "Unisex"],
    "Unisex": ["Unisex", "Men", "Women", "Boys", "Girls"],
}

# Which "slot" of an outfit each clothing type belongs to
SLOT = {}
for t in TOPS:
    SLOT[t] = "top"
for b in BOTTOMS:
    SLOT[b] = "bottom"
for d in DRESSES:
    SLOT[d] = "dress"
for s in SHOES:
    SLOT[s] = "shoe"

# What a sensible outfit looks like: if the user gives us a top, they want
# mostly bottoms plus a couple of shoes - not five pairs of shoes.
OUTFIT_QUOTA = {
    "top": {"bottom": 3, "shoe": 2},
    "bottom": {"top": 3, "shoe": 2},
    "dress": {"shoe": 5},
    "shoe": {"top": 2, "bottom": 2, "dress": 1},
}

# ---------------------------------------------------------------------------
# "Look" (dress style) - DERIVED, not a column in the dataset.
#
# The Kaggle data has no field for cultural or regional dress style, so we do
# not invent one. Instead we derive a practical style label from two real
# columns, `usage` and `articleType`. This is a rule we defined ourselves and
# should be described in the report as a derived feature, not as ground truth.
# ---------------------------------------------------------------------------
LOOK_RULES = [
    # (look name, allowed usages, allowed article types or None for any)
    ("Sportswear / Gym", {"Sports"}, None),
    ("Traditional / Ethnic", {"Ethnic"}, None),
    ("Office / Formal", {"Formal"}, None),
    ("Party / Going out", {"Party"}, None),
    ("Loungewear / Home", {"Casual", "Home"},
     {"Track Pants", "Shorts", "Leggings", "Sweatshirts", "Flats", "Sandals"}),
    ("Smart Casual", {"Smart Casual", "Travel"}, None),
    ("Everyday / Streetwear", {"Casual"}, None),
]


def derive_look(row):
    """Assign one style label to an item using its usage and article type."""
    for look, usages, articles in LOOK_RULES:
        if row["usage"] in usages and (articles is None or row["articleType"] in articles):
            return look
    return "Everyday / Streetwear"


# Colours that go with anything
NEUTRALS = {
    "Black", "White", "Off White", "Grey", "Grey Melange", "Charcoal",
    "Navy Blue", "Beige", "Brown", "Coffee Brown", "Cream", "Silver",
    "Khaki", "Taupe", "Tan", "Nude", "Steel", "Mushroom Brown",
}

# Position of each colour on the colour wheel (like hours on a clock).
# Used to tell analogous colours (neighbours) from clashing ones.
COLOUR_WHEEL = {
    "Red": 0, "Maroon": 0, "Burgundy": 0, "Rust": 1, "Orange": 1, "Peach": 1,
    "Copper": 1, "Bronze": 1, "Gold": 2, "Yellow": 2, "Mustard": 2,
    "Lime Green": 3, "Fluorescent Green": 3, "Olive": 3,
    "Green": 4, "Sea Green": 5, "Teal": 6, "Turquoise Blue": 6,
    "Blue": 7, "Navy Blue": 7, "Lavender": 9, "Purple": 9, "Mauve": 9,
    "Magenta": 10, "Pink": 11, "Rose": 11,
}


def season_ok(s1, s2):
    """Seasons match if they are the same, or in the same warm/cold pair."""
    return s1 == s2 or {"Fall", "Winter"} <= {s1, s2} or {"Spring", "Summer"} <= {s1, s2}


def colour_score(c1, c2):
    """How well two colours work together, from 0.2 (clash) to 0.95 (classic).

    The earlier version returned 1.0 whenever either colour was neutral, which
    made every candidate score the same, so colour stopped affecting the
    ranking at all. This version gives a graded score instead.
    """
    n1, n2 = c1 in NEUTRALS, c2 in NEUTRALS

    if n1 and n2:
        return 0.95 if c1 != c2 else 0.85      # e.g. white + navy: classic
    if n1 or n2:
        return 0.85                            # a neutral with a colour: safe
    if c1 == c2:
        return 0.60                            # all one colour can look flat

    p1, p2 = COLOUR_WHEEL.get(c1), COLOUR_WHEEL.get(c2)
    if p1 is None or p2 is None:
        return 0.50                            # unknown colour: neither good nor bad

    gap = min(abs(p1 - p2), 12 - abs(p1 - p2))  # distance around the wheel
    if gap <= 1:
        return 0.78                            # neighbours: harmonious
    if gap == 2:
        return 0.65
    if gap >= 5:
        return 0.72                            # opposites: deliberate contrast
    return 0.30                                # awkward middle distance: clash


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
    df = df.reset_index(drop=True)
    df["look"] = df.apply(derive_look, axis=1)      # derived style label
    return df


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
        + " " + df["look"].str.replace(r"[ /]", "", regex=True)
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
def find_candidates(df, item, prefs):
    """Candidate items = complementary categories only.

    The filters come from what the USER selected (prefs), not from whichever
    catalogue item we matched them to. Gender is never dropped: relaxing it
    is what previously caused menswear to be suggested for a women's top.
    Occasion and season are relaxed only if too few items remain.
    """
    base = df[df["articleType"].isin(COMPLEMENT[item["articleType"]])]

    allowed_genders = GENDER_GROUPS.get(prefs["gender"], [prefs["gender"], "Unisex"])
    gendered = base[base["gender"].isin(allowed_genders)]
    if gendered.empty:
        gendered = base          # only if the sample has nothing for this gender

    # Style ("look") is applied before occasion, because mixing gym wear with
    # office wear looks worse than being one season out.
    looked = gendered[gendered["look"] == prefs["look"]] if prefs.get("look") else gendered
    if len(looked) < 5:
        looked = gendered

    strict = looked[looked["season"].apply(lambda s: season_ok(s, prefs["season"]))]
    if len(strict) >= 5:
        return strict.copy()

    return looked.copy()


def get_recommendations(item_id, df, cosine_sim, indices, prefs, k=5):
    idx = indices[item_id]
    item = df.iloc[idx]

    candidates = find_candidates(df, item, prefs)
    if candidates.empty:
        return []

    positions = indices[candidates["id"]].values
    style = cosine_sim[idx, positions]
    # colour is compared against what the USER chose, not the matched item
    colour = candidates["baseColour"].apply(lambda c: colour_score(prefs["colour"], c)).values

    candidates["score"] = 0.5 * style + 0.5 * colour
    return balanced_top_k(candidates, item["articleType"], k)


def balanced_top_k(candidates, query_type, k=5):
    """Build a sensible outfit instead of just taking the top K scores.

    For a top we want roughly 3 bottoms and 2 shoes. Taking the raw top K
    lets one group (usually shoes) fill every slot.
    """
    candidates = candidates.copy()
    candidates["slot"] = candidates["articleType"].map(SLOT)
    candidates = candidates.sort_values("score", ascending=False)

    quota = OUTFIT_QUOTA.get(SLOT.get(query_type), {})
    chosen = []
    for slot, count in quota.items():
        picks = candidates[candidates["slot"] == slot]["id"].head(count).tolist()
        chosen.extend(picks)

    # if a slot had too few items, top up with the next best of anything
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
def build_collaborative(df, n_users=None):
    """Simulate shoppers who buy matching pairs together.

    The number of simulated shoppers scales with the catalogue: with a large
    catalogue and too few shoppers, most items are never bought, so
    collaborative filtering has nothing to say about them and returns nothing.
    """
    rng = np.random.default_rng(42)
    anchors = df[df["articleType"].isin(TOPS + DRESSES)]
    if anchors.empty:
        return pd.DataFrame()

    if n_users is None:
        n_users = max(150, len(df) // 2)     # enough coverage for the catalogue

    records = []
    for user in range(n_users):
        for _ in range(12):
            a = anchors.iloc[rng.integers(len(anchors))]
            pool = df[
                df["articleType"].isin(COMPLEMENT[a["articleType"]])
                & (df["look"] == a["look"])
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


def collaborative_recommendations(item_id, df, item_sim, prefs, k=5):
    if item_sim.empty or item_id not in item_sim.index:
        return []

    scores = item_sim[item_id].drop(index=item_id, errors="ignore")
    scores = scores[scores > 0]
    if scores.empty:
        return []

    item = df[df["id"] == item_id].iloc[0]
    allowed_genders = GENDER_GROUPS.get(prefs["gender"], [prefs["gender"], "Unisex"])

    candidates = df[
        df["id"].isin(scores.index)
        & df["articleType"].isin(COMPLEMENT[item["articleType"]])
        & df["gender"].isin(allowed_genders)
    ].copy()

    # keep the outfit in one style, same as the other two methods
    if prefs.get("look"):
        styled = candidates[candidates["look"] == prefs["look"]]
        if len(styled) >= 3:
            candidates = styled

    if candidates.empty:
        return []

    candidates["score"] = candidates["id"].map(scores)
    return balanced_top_k(candidates, item["articleType"], k)


# ---------------------------------------------------------------------------
# Section 5 - Hybrid: average the content score and the collaborative score
# ---------------------------------------------------------------------------
def hybrid_recommendations(item_id, df, cosine_sim, indices, item_sim, prefs, k=5):
    idx = indices[item_id]
    item = df.iloc[idx]

    candidates = find_candidates(df, item, prefs)
    if candidates.empty:
        return []

    positions = indices[candidates["id"]].values
    style = cosine_sim[idx, positions]
    colour = candidates["baseColour"].apply(lambda c: colour_score(prefs["colour"], c)).values
    candidates["content"] = 0.5 * style + 0.5 * colour

    if not item_sim.empty and item_id in item_sim.index:
        candidates["collab"] = candidates["id"].map(item_sim[item_id]).fillna(0)
    else:
        candidates["collab"] = 0.0

    for col in ["content", "collab"]:
        spread = candidates[col].max() - candidates[col].min()
        candidates[col] = (candidates[col] - candidates[col].min()) / spread if spread > 0 else 0

    candidates["score"] = 0.5 * candidates["content"] + 0.5 * candidates["collab"]
    return balanced_top_k(candidates, item["articleType"], k)


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
    """Guess the item's colour from the photo.

    We crop to the middle of the picture and take the median, because the
    average over the whole photo is dominated by the background (a white top
    on a beige table was being read as 'Beige').
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    box = (int(width * 0.25), int(height * 0.25), int(width * 0.75), int(height * 0.75))
    crop = np.array(rgb.crop(box).resize((60, 60))).reshape(-1, 3)

    # ignore very pale pixels (usually background), unless the item itself is pale
    dark_enough = crop.sum(axis=1) < 690
    pixels = crop[dark_enough] if dark_enough.sum() > len(crop) * 0.25 else crop
    middle = np.median(pixels, axis=0)

    options = {c: v for c, v in COLOUR_RGB.items() if c in available}
    if not options:
        return available[0]
    return min(options, key=lambda c: np.linalg.norm(np.array(options[c]) - middle))


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
    usage = st.selectbox("Occasion", sorted(df["usage"].unique()))
    look = st.selectbox("Dress style", sorted(df["look"].unique()))
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
        prefs = {"gender": gender, "usage": usage, "season": season,
                 "colour": colour, "look": look}
        allowed_genders = GENDER_GROUPS.get(gender, [gender, "Unisex"])

        pool = df[df["articleType"] == article_type]
        matching_gender = pool[pool["gender"].isin(allowed_genders)]
        if not matching_gender.empty:
            pool = matching_gender

        if pool.empty:
            st.warning("No products of that type in the sample. Try another type.")
        else:
            pool = pool.copy()
            pool["match"] = (
                (pool["look"] == look).astype(float) * 2      # style matters most
                + (pool["usage"] == usage).astype(float)
                + pool["season"].apply(lambda s: float(season_ok(s, season)))
                + (pool["baseColour"] == colour).astype(float)
            )
            anchor_id = pool.sort_values("match", ascending=False).iloc[0]["id"]

            if method == "Content-based":
                results = get_recommendations(anchor_id, df, cosine_sim, indices, prefs)
            elif method == "Collaborative":
                results = collaborative_recommendations(anchor_id, df, item_sim, prefs)
            else:
                results = hybrid_recommendations(anchor_id, df, cosine_sim, indices, item_sim, prefs)

            if not results:
                st.warning("No matches found with this method. Try Hybrid, or change the filters.")
            else:
                names = df.set_index("id")["productDisplayName"]
                columns = st.columns(len(results))
                for column, result_id in zip(columns, results):
                    path = os.path.join(IMG_DIR, f"{result_id}.jpg")
                    if os.path.exists(path):
                        # fixed width: the source photos are small, so stretching
                        # them across the column makes them look blurry
                        column.image(path, width=170)
                    column.caption(names.get(result_id, str(result_id)))
                st.caption(f"Method: **{method}**")
