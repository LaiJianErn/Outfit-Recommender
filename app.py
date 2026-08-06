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
import colorsys
from collections import Counter

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


def get_recommendations(item_id, df, cosine_sim, indices, prefs, k=5, offset=0):
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
    return balanced_top_k(candidates, item["articleType"], k, offset)


def balanced_top_k(candidates, query_type, k=5, offset=0):
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
        in_slot = candidates[candidates["slot"] == slot]["id"].tolist()
        if not in_slot:
            continue
        # `offset` lets the user ask for another outfit: we step further down
        # the ranked list instead of returning the same items again.
        start = (offset * count) % len(in_slot)
        picks = (in_slot + in_slot)[start:start + count]
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


def collaborative_recommendations(item_id, df, item_sim, prefs, k=5, offset=0):
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
    return balanced_top_k(candidates, item["articleType"], k, offset)


# ---------------------------------------------------------------------------
# Section 5 - Hybrid: average the content score and the collaborative score
# ---------------------------------------------------------------------------
def hybrid_recommendations(item_id, df, cosine_sim, indices, item_sim, prefs, k=5, offset=0):
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
    return balanced_top_k(candidates, item["articleType"], k, offset)


# ---------------------------------------------------------------------------
# Helper: detect the main colour of an uploaded photo
# ---------------------------------------------------------------------------
# Hue anchors around the colour wheel, used to name the colour in a photo.
# Comparing raw RGB distance does not work: a pale pink is numerically closer
# to Silver and Beige than to Pink, which is why light garments were being
# misread. Hue survives lightness changes, so we classify on hue instead.
HUE_NAMES = [
    (0, "Red"), (20, "Orange"), (45, "Yellow"), (75, "Lime Green"),
    (120, "Green"), (160, "Sea Green"), (180, "Teal"), (215, "Blue"),
    (270, "Purple"), (300, "Magenta"), (330, "Pink"), (355, "Red"),
]


def classify_pixel(rgb):
    """Name a single pixel's colour using hue, saturation and brightness."""
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360

    if s < 0.12:                                   # barely any colour: a neutral
        if v < 0.18:
            return "Black"
        if v < 0.45:
            return "Charcoal"
        if v < 0.68:
            return "Grey"
        if v < 0.88:
            return "Silver"
        return "White"

    name = min(HUE_NAMES, key=lambda x: min(abs(hue - x[0]), 360 - abs(hue - x[0])))[1]

    if name == "Red":
        if v < 0.45:
            return "Maroon"
        if s < 0.35 and v > 0.75:
            return "Pink"                          # a pale tint of red is pink
    if name == "Orange" and v < 0.5:
        return "Brown"
    if name == "Yellow" and s < 0.3:
        return "Beige"
    if name == "Blue" and v < 0.4:
        return "Navy Blue"
    if name == "Pink" and v < 0.5:
        return "Maroon"
    return name


def detect_colour(image, available):
    """Guess the garment's colour from the photo.

    Every pixel in the middle of the picture is named, then we take the most
    common name. Plain white and silver are skipped first because they are
    usually the background, not the item.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    box = (int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.8))
    pixels = np.array(rgb.crop(box).resize((80, 80))).reshape(-1, 3)

    counts = Counter(classify_pixel(p) for p in pixels)
    ranked = [name for name, _ in counts.most_common()]

    for name in ranked:                            # prefer an actual colour
        if name not in ("White", "Silver") and name in available:
            return name
    for name in ranked:                            # otherwise whatever we have
        if name in available:
            return name
    return available[0]


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Styling. Ink-blue and warm sand, set in Fraunces (a high-contrast display
# face used only for the wordmark) over Inter for everything else. The rule
# under each method name is the structural device: it separates the three
# recommenders without adding boxes.
# ---------------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  .stApp { background: #FBF9F5; }
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1C2B3A; }

  .wordmark {
      font-family: 'Fraunces', serif; font-size: 2.6rem; font-weight: 600;
      color: #1C2B3A; letter-spacing: -0.02em; line-height: 1.05; margin: 0;
  }
  .tagline {
      font-size: 0.9rem; color: #7A8794; margin: 0.3rem 0 0.2rem 0;
      letter-spacing: 0.01em;
  }
  .count { font-size: 0.78rem; color: #A2ADB8; letter-spacing: 0.06em;
           text-transform: uppercase; }

  .step {
      font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
      text-transform: uppercase; color: #C08552; margin: 0.4rem 0 0.1rem 0;
  }

  .method-head { margin: 1.6rem 0 0.9rem 0; }
  .method-name {
      font-family: 'Fraunces', serif; font-size: 1.35rem; color: #1C2B3A;
      margin-right: 0.7rem;
  }
  .method-blurb { font-size: 0.85rem; color: #7A8794; }

  .item-name {
      font-size: 0.8rem; font-weight: 500; color: #1C2B3A; line-height: 1.3;
      margin-top: 0.45rem;
  }
  .item-meta { font-size: 0.72rem; color: #A2ADB8; margin-top: 0.1rem; }

  .divider { height: 1px; background: #E8E2D8; margin: 1.5rem 0 0.2rem 0; }

  .placeholder {
      border: 1px dashed #DDD5C8; border-radius: 4px; padding: 2.5rem 2rem;
      color: #7A8794; font-size: 0.92rem; line-height: 1.6; text-align: center;
      margin-top: 1rem;
  }

  div.stButton > button[kind="primary"] {
      background: #1C2B3A; color: #FBF9F5; border: none; border-radius: 3px;
      font-weight: 500; letter-spacing: 0.02em;
  }
  div.stButton > button[kind="primary"]:hover { background: #2E4257; }
  div.stButton > button[kind="secondary"] {
      background: transparent; color: #1C2B3A; border: 1px solid #DDD5C8;
      border-radius: 3px; font-weight: 500;
  }
  img { border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='wordmark'>Outfit Recommender</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='tagline'>Upload a piece you own. Three recommender methods "
    "each suggest what completes the look.</div>",
    unsafe_allow_html=True,
)

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

@st.cache_data
def image_display_width(ids):
    """Pick a display size that suits the photos we actually have.

    Small source photos are shown smaller so they stay sharp; larger photos are
    shown bigger. This means swapping in the high-resolution dataset improves
    the display automatically, with no code change.
    """
    widths = []
    for i in list(ids)[:25]:
        path = os.path.join(IMG_DIR, f"{i}.jpg")
        if os.path.exists(path):
            with Image.open(path) as im:
                widths.append(im.size[0])
    if not widths:
        return 140
    typical = sorted(widths)[len(widths) // 2]
    return int(min(180, max(110, typical * 2)))


DISPLAY_WIDTH = image_display_width(df["id"].tolist())

cosine_sim = build_similarity(df)
indices = pd.Series(df.index, index=df["id"])     # reverse map, like the practical
item_sim = build_collaborative(df)

_res = "low-res" if DISPLAY_WIDTH < 150 else "high-res"
st.markdown(
    f"<div class='count'>{len(df)} products in catalogue &middot; {_res} images</div>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 2])

with left:
    st.markdown("<div class='step'>Step 1 &mdash; Your item</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Photo of a top, bottom, dress or shoe", type=["jpg", "jpeg", "png"])

    detected = None
    if uploaded:
        photo = Image.open(uploaded)
        st.image(photo, caption="Your upload", width='stretch')
        detected = detect_colour(photo, sorted(df["baseColour"].unique()))
        st.success(f"Detected colour: **{detected}**")

    st.markdown("<div class='step'>Step 2 &mdash; Confirm details</div>", unsafe_allow_html=True)
    group = st.selectbox("Item type", ["Top", "Bottom", "Dress", "Shoes"])
    options = {"Top": TOPS, "Bottom": BOTTOMS, "Dress": DRESSES, "Shoes": SHOES}[group]
    options = [t for t in options if t in set(df["articleType"])] or options
    article_type = st.selectbox("Specific type", options)

    colours = sorted(df["baseColour"].unique())
    colour_index = colours.index(detected) if detected in colours else 0
    colour = st.selectbox("Colour", colours, index=colour_index)

    gender_options = sorted(df["gender"].unique())
    default_gender = gender_options.index("Women") if "Women" in gender_options else 0
    gender = st.selectbox(
        "Gender", gender_options, index=default_gender,
        help="Not read from the photo - please set this yourself.",
    )
    usage = st.selectbox("Occasion", sorted(df["usage"].unique()))
    look = st.selectbox("Dress style", sorted(df["look"].unique()))
    season = st.selectbox("Season", sorted(df["season"].unique()))
<<<<<<< HEAD
    search = st.button("Find matching outfits", type="primary")
=======
    method = st.radio("Method", ["Hybrid", "Content-based", "Collaborative"])
    search = st.button("Find matching items", type="primary")
>>>>>>> a5bdb411a79fe54a3460d61b16b0e5b052d97666
    if st.button("Show me another option"):
        st.session_state["variant"] = st.session_state.get("variant", 0) + 1
        search = True
    variant = st.session_state.get("variant", 0)

with right:
    if not search:
        st.markdown(
            "<div class='placeholder'>Set the details on the left, then press "
            "<b>Find matching outfits</b>.<br>All three recommender methods will "
            "be shown side by side for comparison.</div>",
            unsafe_allow_html=True,
        )
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

<<<<<<< HEAD
            names = df.set_index("id")["productDisplayName"]
            colours = df.set_index("id")["baseColour"]
            types = df.set_index("id")["articleType"]
=======
            if method == "Content-based":
                results = get_recommendations(anchor_id, df, cosine_sim, indices, prefs, offset=variant)
            elif method == "Collaborative":
                results = collaborative_recommendations(anchor_id, df, item_sim, prefs, offset=variant)
            else:
                results = hybrid_recommendations(anchor_id, df, cosine_sim, indices, item_sim, prefs, offset=variant)
>>>>>>> a5bdb411a79fe54a3460d61b16b0e5b052d97666

            methods = [
                ("Content-based", "Item attributes: colour, style, season",
                 lambda: get_recommendations(anchor_id, df, cosine_sim, indices,
                                             prefs, offset=variant)),
                ("Collaborative", "What similar shoppers bought together",
                 lambda: collaborative_recommendations(anchor_id, df, item_sim,
                                                       prefs, offset=variant)),
                ("Hybrid", "Both signals combined, 50/50",
                 lambda: hybrid_recommendations(anchor_id, df, cosine_sim, indices,
                                                item_sim, prefs, offset=variant)),
            ]

            for name, blurb, fn in methods:
                results = fn()
                st.markdown(
                    f"<div class='method-head'><span class='method-name'>{name}</span>"
                    f"<span class='method-blurb'>{blurb}</span></div>",
                    unsafe_allow_html=True,
                )
                if not results:
                    st.warning("No matches with this method. Try different filters.")
                    continue

                columns = st.columns(5)
                for column, result_id in zip(columns, results):
                    path = os.path.join(IMG_DIR, f"{result_id}.jpg")
                    with column:
                        if os.path.exists(path):
                            st.image(path, width=DISPLAY_WIDTH)
                        st.markdown(
                            f"<div class='item-name'>{names.get(result_id, result_id)}</div>"
                            f"<div class='item-meta'>{types.get(result_id, '')} &middot; "
                            f"{colours.get(result_id, '')}</div>",
                            unsafe_allow_html=True,
                        )
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
