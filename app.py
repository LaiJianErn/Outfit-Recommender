"""
Outfit Recommender — Streamlit app
Deploy via GitHub + Streamlit Community Cloud (share.streamlit.io).

Runs on a built-in SYNTHETIC catalogue out of the box (no setup needed).
Synthetic items are rendered as simple garment-silhouette icons (top/pants/
skirt/dress/shoe shapes) in the item's colour, so results look like
recognizable clothing rather than plain colour blocks.

To use REAL Kaggle photos: add a folder named `sample_data/` next to this file,
containing `styles.csv` + an `images/` folder (a subset exported from your
Colab notebook). The app auto-detects it and switches over automatically.
"""
import os, glob, random
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42); np.random.seed(42)
st.set_page_config(page_title="Outfit Recommender", layout="wide")

# ----------------------------------------------------------------------------
# DOMAIN RULES
# ----------------------------------------------------------------------------
TOPS    = ['Shirts', 'Tshirts', 'Tops', 'Sweaters', 'Sweatshirts', 'Kurtas']
BOTTOMS = ['Jeans', 'Trousers', 'Skirts', 'Shorts', 'Track Pants', 'Leggings']
DRESSES = ['Dresses']
SHOES   = ['Casual Shoes', 'Heels', 'Sandals', 'Sports Shoes', 'Flats', 'Formal Shoes']
ALL_TYPES = TOPS + BOTTOMS + DRESSES + SHOES

COMPLEMENT = {}
for t in TOPS:    COMPLEMENT[t] = BOTTOMS + SHOES
for b in BOTTOMS: COMPLEMENT[b] = TOPS + SHOES
for d in DRESSES: COMPLEMENT[d] = SHOES
for s in SHOES:   COMPLEMENT[s] = TOPS + BOTTOMS + DRESSES

SLOT = {}
for t in TOPS:    SLOT[t] = 'top'
for b in BOTTOMS: SLOT[b] = 'bottom'
for d in DRESSES: SLOT[d] = 'dress'
for s in SHOES:   SLOT[s] = 'shoe'

NEUTRALS = {'Black','White','Grey','Navy Blue','Beige','Brown','Cream'}
HARMONY = [
    {'Blue','Navy Blue','White','Grey'}, {'Red','Black','White','Maroon'},
    {'Pink','White','Grey','Maroon'}, {'Green','Beige','Brown','White'},
    {'Yellow','Blue','White','Grey'},
]
def colour_score(c1, c2):
    if c1 in NEUTRALS or c2 in NEUTRALS: return 1.0
    if c1 == c2: return 0.8
    for g in HARMONY:
        if c1 in g and c2 in g: return 0.7
    return 0.2

def season_ok(s1, s2):
    return (s1==s2) or ({'Fall','Winter'} <= {s1,s2}) or ({'Spring','Summer'} <= {s1,s2})

# rough RGB anchors for dominant-colour detection from an uploaded photo
COLOUR_ANCHORS = {
    'Black':(20,20,20), 'White':(245,245,245), 'Grey':(150,150,150),
    'Navy Blue':(30,40,80), 'Blue':(50,100,190), 'Red':(190,40,40),
    'Maroon':(110,30,45), 'Green':(50,140,80), 'Olive':(110,110,40),
    'Yellow':(225,200,60), 'Beige':(220,205,175), 'Brown':(110,70,40),
    'Pink':(230,140,170), 'Purple':(110,60,140), 'Orange':(220,120,40),
    'Silver':(192,192,192), 'Cream':(245,240,215),
}

def detect_dominant_colour(img: Image.Image, available_colours):
    small = np.array(img.convert('RGB').resize((50, 50))).reshape(-1, 3).mean(axis=0)
    options = {k: v for k, v in COLOUR_ANCHORS.items() if k in available_colours}
    if not options:
        return available_colours[0]
    best = min(options, key=lambda k: np.linalg.norm(np.array(options[k]) - small))
    return best

# ----------------------------------------------------------------------------
# DATA LOADING (cached so it only runs once per app session)
# ----------------------------------------------------------------------------
COLS = ['id','gender','masterCategory','subCategory','articleType',
        'baseColour','season','year','usage','productDisplayName']

def icon_kind_for(article_type):
    """Maps an articleType to which silhouette shape to draw."""
    if article_type == 'Skirts': return 'skirt'
    if article_type in BOTTOMS: return 'pants'
    if article_type in TOPS: return 'top'
    if article_type in DRESSES: return 'dress'
    if article_type == 'Heels': return 'heel'
    if article_type == 'Sandals': return 'sandal'
    if article_type in SHOES: return 'shoe'
    return 'shoe'

def draw_garment_icon(kind, color_rgb, size=(160, 200)):
    """Draws a simple, recognizable garment silhouette filled with color_rgb,
    so placeholder items look like clothing rather than plain rectangles."""
    img = Image.new('RGB', size, (250, 250, 250))
    d = ImageDraw.Draw(img)
    w, h = size
    outline = (70, 70, 70)

    if kind == 'top':
        bt, bb = h*0.30, h*0.85
        bl, br = w*0.28, w*0.72
        d.polygon([(bl, bt), (w*0.04, h*0.42), (w*0.04, h*0.58), (bl, bt+h*0.14)], fill=color_rgb, outline=outline)
        d.polygon([(br, bt), (w*0.96, h*0.42), (w*0.96, h*0.58), (br, bt+h*0.14)], fill=color_rgb, outline=outline)
        d.rectangle([bl, bt, br, bb], fill=color_rgb, outline=outline)
        d.ellipse([w*0.40, bt-h*0.05, w*0.60, bt+h*0.06], fill=(250,250,250), outline=outline)
    elif kind == 'pants':
        ty, by = h*0.16, h*0.88
        wl, wr = w*0.30, w*0.70
        mid = w*0.50
        d.polygon([(wl, ty), (mid-3, ty), (mid-7, by), (w*0.34, by)], fill=color_rgb, outline=outline)
        d.polygon([(mid+3, ty), (wr, ty), (w*0.66, by), (mid+7, by)], fill=color_rgb, outline=outline)
    elif kind == 'skirt':
        d.polygon([(w*0.35, h*0.20), (w*0.65, h*0.20), (w*0.82, h*0.82), (w*0.18, h*0.82)], fill=color_rgb, outline=outline)
    elif kind == 'dress':
        d.polygon([(w*0.38, h*0.18), (w*0.62, h*0.18), (w*0.80, h*0.85), (w*0.20, h*0.85)], fill=color_rgb, outline=outline)
        d.polygon([(w*0.38, h*0.18), (w*0.14, h*0.32), (w*0.24, h*0.40), (w*0.38, h*0.29)], fill=color_rgb, outline=outline)
        d.polygon([(w*0.62, h*0.18), (w*0.86, h*0.32), (w*0.76, h*0.40), (w*0.62, h*0.29)], fill=color_rgb, outline=outline)
        d.ellipse([w*0.42, h*0.14, w*0.58, h*0.23], fill=(250,250,250), outline=outline)
    elif kind == 'heel':
        d.polygon([(w*0.18, h*0.62), (w*0.75, h*0.55), (w*0.85, h*0.68), (w*0.20, h*0.72)], fill=color_rgb, outline=outline)
        d.polygon([(w*0.75, h*0.68), (w*0.80, h*0.90), (w*0.68, h*0.90), (w*0.68, h*0.70)], fill=color_rgb, outline=outline)
        d.polygon([(w*0.18, h*0.62), (w*0.22, h*0.45), (w*0.55, h*0.42), (w*0.75, h*0.55)], fill=color_rgb, outline=outline)
    elif kind == 'sandal':
        d.ellipse([w*0.15, h*0.68, w*0.85, h*0.82], fill=color_rgb, outline=outline)
        d.line([(w*0.50, h*0.68), (w*0.35, h*0.45), (w*0.65, h*0.45), (w*0.50, h*0.68)], fill=outline, width=3)
    else:  # generic shoe
        d.ellipse([w*0.10, h*0.60, w*0.90, h*0.80], fill=color_rgb, outline=outline)
        d.polygon([(w*0.15, h*0.64), (w*0.18, h*0.40), (w*0.55, h*0.35), (w*0.75, h*0.60)], fill=color_rgb, outline=outline)
    return img

def make_synthetic(n=450, img_dir='synthetic_images'):
    os.makedirs(img_dir, exist_ok=True)
    rgb = {'Black':(30,30,30),'White':(240,240,240),'Grey':(150,150,150),
           'Navy Blue':(40,50,90),'Beige':(225,210,180),'Brown':(120,80,50),
           'Cream':(250,245,225),'Blue':(60,110,200),'Red':(200,50,50),
           'Green':(60,150,90),'Pink':(235,150,180),'Yellow':(235,210,70),'Maroon':(120,40,55)}
    colours = list(rgb.keys()); genders=['Men','Women','Unisex']
    seasons=['Summer','Winter','Fall','Spring']; usages=['Casual','Formal','Sports','Ethnic']
    rows = []
    for i in range(n):
        at = random.choice(ALL_TYPES)
        sub = ('Topwear' if at in TOPS else 'Bottomwear' if at in BOTTOMS
               else 'Dress' if at in DRESSES else 'Shoes')
        mc = 'Footwear' if at in SHOES else 'Apparel'
        col = random.choice(colours); g = random.choice(genders)
        pid = 100000 + i
        rows.append([pid, g, mc, sub, at, col, random.choice(seasons), 2018,
                     random.choice(usages), f"{g} {col} {at}"])
        icon = draw_garment_icon(icon_kind_for(at), rgb[col])
        icon.save(f"{img_dir}/{pid}.jpg")
    return pd.DataFrame(rows, columns=COLS), img_dir

@st.cache_data(show_spinner="Loading catalogue...")
def load_catalog():
    hits = glob.glob('sample_data/**/styles.csv', recursive=True)
    for h in hits:
        img = os.path.join(os.path.dirname(h), 'images')
        if os.path.isdir(img):
            df = pd.read_csv(h, on_bad_lines='skip')
            df = df[df['articleType'].isin(COMPLEMENT.keys())].copy()
            df = df.dropna(subset=['gender','articleType','baseColour','season','usage']).reset_index(drop=True)
            return df, img, False
    df, img = make_synthetic()
    return df, img, True

@st.cache_data(show_spinner="Building similarity model...")
def build_models(df):
    soup = (df['gender'] + ' ' + df['usage'] + ' ' + df['season'] + ' '
            + df['baseColour'].str.replace(' ', '') + ' ' + df['articleType'])
    tfidf = TfidfVectorizer()
    mat = tfidf.fit_transform(soup)
    cos = cosine_similarity(mat, mat)

    # simulate outfits + purchases (same logic as the main notebook) for
    # collaborative filtering + ground truth
    def compatible(a, b):
        if b['articleType'] not in COMPLEMENT.get(a['articleType'], []): return False
        if not (a['gender']==b['gender'] or 'Unisex' in (a['gender'],b['gender'])): return False
        if a['usage'] != b['usage']: return False
        if not season_ok(a['season'], b['season']): return False
        return colour_score(a['baseColour'], b['baseColour']) >= 0.5

    anchors = df.index[df['articleType'].isin(TOPS+DRESSES)].tolist()
    outfits = []
    for anc in anchors:
        a = df.loc[anc]
        cand = df[df['articleType'].isin(COMPLEMENT[a['articleType']])]
        for _, b in cand.sample(min(30, len(cand)), random_state=anc).iterrows():
            if compatible(a, b): outfits.append((a['id'], b['id']))
    outfits = list(set(outfits))

    inter = []
    for u in range(60):
        for _ in range(random.randint(8, 18)):
            if outfits:
                x, y = random.choice(outfits); inter += [(u, x), (u, y)]
    inter_df = pd.DataFrame(inter, columns=['user_id','item_id']).drop_duplicates()
    inter_df['rating'] = 1
    ui = inter_df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    item_sim_df = pd.DataFrame(cosine_similarity(ui.T), index=ui.columns, columns=ui.columns) if not ui.empty else pd.DataFrame()
    return cos, tfidf, item_sim_df

# ----------------------------------------------------------------------------
# RECOMMENDER LOGIC (with diversify fix: guarantees representation per slot)
# ----------------------------------------------------------------------------
def diversify(cand_df, score_col, k=5):
    cand_df = cand_df.copy()
    cand_df['slot'] = cand_df['articleType'].map(SLOT)
    cand_df = cand_df.sort_values(score_col, ascending=False)
    chosen = []
    for slot in cand_df['slot'].dropna().unique():
        sub = cand_df[cand_df['slot'] == slot]
        if not sub.empty:
            chosen.append(sub.iloc[0]['id'])
    for _, row in cand_df.iterrows():
        if len(chosen) >= k: break
        if row['id'] not in chosen:
            chosen.append(row['id'])
    return chosen[:k]

def complete_the_look(df, cos, id_to_pos, item_id, k=5):
    i = id_to_pos[item_id]; a = df.iloc[i]
    cand = df[df['articleType'].isin(COMPLEMENT[a['articleType']])].copy()
    cand = cand[(cand['gender'].isin([a['gender'],'Unisex'])) | (a['gender']=='Unisex')]
    cand = cand[cand['usage'] == a['usage']]
    cand = cand[cand['season'].apply(lambda s: season_ok(s, a['season']))]
    if cand.empty: return []
    cand_idx = cand['id'].map(id_to_pos).values
    cand['style']  = cos[i, cand_idx]
    cand['colour'] = cand['baseColour'].apply(lambda c: colour_score(a['baseColour'], c))
    cand['final']  = 0.5*cand['style'] + 0.5*cand['colour']
    return diversify(cand, 'final', k)

def collaborative_recommend(df, item_sim_df, item_id, k=5):
    if item_sim_df.empty or item_id not in item_sim_df.index:
        return []
    sims = item_sim_df[item_id].drop(index=item_id, errors='ignore')
    a = df.set_index('id').loc[item_id]
    art = df.set_index('id')['articleType']
    keep = [iid for iid in sims.index if art.get(iid) in COMPLEMENT[a['articleType']]]
    sims = sims.loc[keep]; sims = sims[sims > 0]
    if sims.empty: return []
    cand = df[df['id'].isin(sims.index)].copy()
    cand['final'] = cand['id'].map(sims)
    return diversify(cand, 'final', k)

def hybrid_recommend(df, cos, item_sim_df, id_to_pos, item_id, k=5, w_content=0.5):
    i = id_to_pos[item_id]; a = df.iloc[i]
    cand = df[df['articleType'].isin(COMPLEMENT[a['articleType']])].copy()
    cand = cand[cand['usage'] == a['usage']]
    if cand.empty: return []
    cand_idx = cand['id'].map(id_to_pos).values
    style  = cos[i, cand_idx]
    colour = cand['baseColour'].apply(lambda c: colour_score(a['baseColour'], c)).values
    cand['content'] = 0.5*style + 0.5*colour
    if not item_sim_df.empty and item_id in item_sim_df.index:
        cand['collab'] = cand['id'].map(item_sim_df[item_id]).fillna(0).clip(lower=0)
    else:
        cand['collab'] = 0
    for c in ['content','collab']:
        rng = cand[c].max() - cand[c].min()
        cand[c] = (cand[c]-cand[c].min())/rng if rng > 0 else 0
    cand['final'] = w_content*cand['content'] + (1-w_content)*cand['collab']
    return diversify(cand, 'final', k)

def find_nearest_item(df, article_type, gender, usage, season, colour):
    """Snap an uploaded photo's profile to the closest existing catalogue item,
    so we have a real anchor id to run the recommenders on (handles the
    cold-start case: a brand-new uploaded item has no history of its own)."""
    cand = df[df['articleType'] == article_type]
    gender_match = cand[cand['gender'].isin([gender, 'Unisex'])]
    if not gender_match.empty:
        cand = gender_match
    if cand.empty:
        return None
    def score(row):
        s = (1.0 if row['usage'] == usage else 0.0)
        s += (1.0 if season_ok(row['season'], season) else 0.0)
        s += colour_score(row['baseColour'], colour)
        return s
    cand = cand.copy()
    cand['match'] = cand.apply(score, axis=1)
    return cand.sort_values('match', ascending=False).iloc[0]['id']

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
df, IMG_DIR, IS_SYNTH = load_catalog()
id_to_pos = pd.Series(df.index, index=df['id'])
cos, tfidf, item_sim_df = build_models(df)

st.title("👕 Outfit Recommender")
st.caption(f"Catalogue: {len(df)} items \u00b7 {'synthetic demo data' if IS_SYNTH else 'real Kaggle data'}")

left, right = st.columns([1, 2])

with left:
    st.subheader("1. Upload your item")
    uploaded = st.file_uploader("Photo of a top, bottom, dress or shoe", type=['jpg','jpeg','png'])
    detected_colour = None
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Your upload", use_container_width=True)
        detected_colour = detect_dominant_colour(img, sorted(df['baseColour'].unique()))
        st.success(f"Detected dominant colour: **{detected_colour}**")

    st.subheader("2. Confirm details")
    group = st.selectbox("What type of item is this?", ["Top", "Bottom", "Dress", "Shoes"])
    type_options = {"Top": TOPS, "Bottom": BOTTOMS, "Dress": DRESSES, "Shoes": SHOES}[group]
    type_options = [t for t in type_options if t in df['articleType'].unique()] or type_options
    article_type = st.selectbox("Specific type", type_options)

    colour_list = sorted(df['baseColour'].unique())
    default_idx = colour_list.index(detected_colour) if detected_colour in colour_list else 0
    colour = st.selectbox("Colour (auto-filled from photo \u2014 override if wrong)", colour_list, index=default_idx)

    gender = st.selectbox("Gender", sorted(df['gender'].unique()))
    usage  = st.selectbox("Style / occasion", sorted(df['usage'].unique()))
    season = st.selectbox("Season", sorted(df['season'].unique()))
    method = st.radio("Recommender method", ["Hybrid", "Content-based", "Collaborative"])
    go = st.button("✨ Find matching outfit pieces", type="primary")

with right:
    st.subheader("Recommended outfit")
    if go:
        anchor_id = find_nearest_item(df, article_type, gender, usage, season, colour)
        if anchor_id is None:
            st.warning("No catalogue item matches that combination yet \u2014 try different filters.")
        else:
            if method == "Content-based":
                rec_ids = complete_the_look(df, cos, id_to_pos, anchor_id, k=5)
            elif method == "Collaborative":
                rec_ids = collaborative_recommend(df, item_sim_df, anchor_id, k=5)
            else:
                rec_ids = hybrid_recommend(df, cos, item_sim_df, id_to_pos, anchor_id, k=5)

            if not rec_ids:
                st.warning("No recommendations found for this combination with this method \u2014 try Hybrid or different filters.")
            else:
                name = df.set_index('id')['productDisplayName']
                cols = st.columns(len(rec_ids))
                for c, rid in zip(cols, rec_ids):
                    p = f"{IMG_DIR}/{rid}.jpg"
                    if os.path.exists(p):
                        c.image(p, use_container_width=True)
                    c.caption(name.get(rid, str(rid)))
                st.caption(f"Method used: **{method}**")
    else:
        st.info("Upload a photo and click the button to see recommendations here.")
