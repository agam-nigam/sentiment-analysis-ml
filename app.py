"""
Sentiment Analysis Web App
--------------------------
TF-IDF + Linear SVM sentiment / emotion classifier.

Run with:  streamlit run app.py
Files required in the same folder:
    best_sentiment_model.pkl
    tfidf_vectorizer.pkl
    cleaned_sentiment_dataset.csv
"""

import re
import string

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Sentiment Analyser | ML Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. STYLES
# =========================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root{
    --bg:#0B0E14;
    --surface:#141922;
    --surface-2:#1B2130;
    --border:rgba(255,255,255,.07);
    --text:#E8ECF3;
    --muted:#8B95A7;
    --accent:#6C8CFF;
    --accent-2:#9B6CFF;
    --pos:#2ED47A;
    --neg:#FF5C6C;
    --neu:#FFC53D;
}

html, body, [class*="css"], .stApp{ font-family:'Inter', sans-serif; }

.stApp{
    background:
        radial-gradient(1000px 500px at 12% -10%, rgba(108,140,255,.16), transparent 60%),
        radial-gradient(900px 500px at 88% 0%, rgba(155,108,255,.13), transparent 55%),
        var(--bg);
    color:var(--text);
}

#MainMenu, footer, header {visibility:hidden;}

.block-container{padding-top:2.2rem; padding-bottom:3rem; max-width:1180px;}

/* ---------- Hero ---------- */
.hero{
    border:1px solid var(--border);
    border-radius:22px;
    padding:38px 34px;
    background:linear-gradient(145deg, rgba(108,140,255,.10), rgba(155,108,255,.05));
    margin-bottom:26px;
}
.hero h1{
    font-size:42px; font-weight:800; letter-spacing:-1px; margin:0 0 10px 0;
    background:linear-gradient(90deg,#FFFFFF 10%, var(--accent) 55%, var(--accent-2) 95%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p{color:var(--muted); font-size:16px; margin:0; max-width:640px; line-height:1.6;}

.pill{
    display:inline-block; padding:6px 14px; margin:14px 8px 0 0;
    font-size:12.5px; font-weight:500; letter-spacing:.3px; color:#C9D3E6;
    border:1px solid var(--border); border-radius:999px; background:rgba(255,255,255,.04);
}

/* ---------- Cards ---------- */
.card{
    background:var(--surface); border:1px solid var(--border); border-radius:18px;
    padding:24px 26px; box-shadow:0 10px 34px rgba(0,0,0,.35); margin-bottom:18px;
}
.card h3{
    font-size:15px; font-weight:600; color:var(--muted);
    text-transform:uppercase; letter-spacing:1.4px; margin:0 0 14px 0;
}

/* ---------- Result banner ---------- */
.result{
    border-radius:18px; padding:26px; text-align:center;
    border:1px solid var(--border); margin-bottom:6px;
}
.result .label{font-size:34px; font-weight:800; letter-spacing:-.5px; margin:6px 0;}
.result .sub{font-size:13px; color:var(--muted); letter-spacing:1.5px; text-transform:uppercase;}
.r-pos{background:linear-gradient(160deg, rgba(46,212,122,.20), rgba(46,212,122,.04));}
.r-pos .label{color:var(--pos);}
.r-neg{background:linear-gradient(160deg, rgba(255,92,108,.20), rgba(255,92,108,.04));}
.r-neg .label{color:var(--neg);}
.r-neu{background:linear-gradient(160deg, rgba(255,197,61,.20), rgba(255,197,61,.04));}
.r-neu .label{color:var(--neu);}

/* ---------- Stat tiles ---------- */
.stat{
    background:var(--surface-2); border:1px solid var(--border);
    border-radius:14px; padding:16px 18px; text-align:center;
}
.stat .v{font-size:26px; font-weight:700; color:var(--text);}
.stat .k{font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:1.2px;}

/* ---------- Inputs / buttons ---------- */
.stTextArea textarea{
    background:var(--surface-2)!important; color:var(--text)!important;
    border:1px solid var(--border)!important; border-radius:14px!important;
    font-size:15px!important; padding:14px!important;
}
.stTextArea textarea:focus{
    border-color:var(--accent)!important;
    box-shadow:0 0 0 3px rgba(108,140,255,.18)!important;
}
.stButton>button{
    width:100%; border-radius:12px; border:1px solid var(--border);
    background:var(--surface-2); color:var(--text); font-weight:600;
    padding:.62rem 1rem; transition:all .18s ease;
}
.stButton>button:hover{border-color:var(--accent); color:#FFFFFF; transform:translateY(-1px);}
.stButton>button[kind="primary"]{
    background:linear-gradient(90deg, var(--accent), var(--accent-2));
    border:none; color:#fff;
}
.stProgress > div > div > div > div{
    background:linear-gradient(90deg, var(--accent), var(--accent-2));
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{ background:#0E121B; border-right:1px solid var(--border); }
section[data-testid="stSidebar"] .side-title{font-size:19px; font-weight:700; color:var(--text); margin-bottom:2px;}
section[data-testid="stSidebar"] .side-sub{font-size:12px; color:var(--muted); margin-bottom:18px;}
.badge{
    display:flex; align-items:center; gap:8px; font-size:12.5px; color:var(--muted);
    border:1px solid var(--border); border-radius:10px; padding:9px 12px;
    margin-bottom:8px; background:rgba(255,255,255,.03);
}
.dot{width:8px; height:8px; border-radius:50%; background:var(--pos); box-shadow:0 0 10px var(--pos);}

/* ---------- Misc ---------- */
.stDataFrame{border-radius:14px; overflow:hidden;}
.foot{text-align:center; color:var(--muted); font-size:12.5px; padding:24px 0 6px;}
hr{border-color:var(--border);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================================================
# 3. LOADERS
# =========================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("best_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


@st.cache_data(show_spinner=False)
def load_data():
    data = pd.read_csv("cleaned_sentiment_dataset.csv")
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()
    return data


MODEL_OK = True
LOAD_ERROR = ""
try:
    model, vectorizer = load_artifacts()
    df = load_data()
except Exception as exc:                       # noqa: BLE001
    MODEL_OK = False
    LOAD_ERROR = str(exc)

# =========================================================
# 4. LABEL -> POLARITY MAPPING
# The trained model returns fine-grained emotion labels
# (Joy, Grief, Curiosity ...), so they are grouped into
# three polarities for a clean verdict in the UI.
# =========================================================
POSITIVE_WORDS = {
    "positive", "joy", "happy", "happiness", "excitement", "elation", "euphoria",
    "contentment", "gratitude", "grateful", "serenity", "hope", "hopeful",
    "love", "affection", "admiration", "adoration", "pride", "proud", "awe",
    "enthusiasm", "inspiration", "inspired", "amusement", "playful", "kind",
    "compassion", "compassionate", "empathetic", "tenderness", "thrill", "zest",
    "satisfaction", "fulfillment", "accomplishment", "acceptance", "calmness",
    "coziness", "creativity", "curiosity", "captivation", "enchantment",
    "enjoyment", "empowerment", "determination", "confident", "adventure",
    "exploration", "rejuvenation", "tranquility", "reverence", "free-spirited",
    "whimsy", "mischievous", "anticipation", "nostalgia",
}
NEGATIVE_WORDS = {
    "negative", "sad", "sadness", "anger", "hate", "fear", "fearful", "grief",
    "despair", "desolation", "loneliness", "isolation", "loss", "regret",
    "shame", "embarrassed", "disgust", "bitter", "bitterness", "betrayal",
    "jealous", "jealousy", "envy", "envious", "resentment", "frustration",
    "frustrated", "disappointment", "disappointed", "devastated", "heartbreak",
    "melancholy", "numbness", "boredom", "apprehensive", "overwhelmed",
    "dismissive", "bad", "yearning",
}
NEUTRAL_WORDS = {
    "neutral", "other", "reflection", "contemplation", "ambivalence",
    "indifference", "emotion", "surprise", "confusion", "arousal",
}

POLARITY_META = {
    "Positive": ("😊", "r-pos"),
    "Negative": ("😞", "r-neg"),
    "Neutral":  ("😐", "r-neu"),
}


def to_polarity(label) -> str:
    key = str(label).strip().lower()
    if key in POSITIVE_WORDS:
        return "Positive"
    if key in NEGATIVE_WORDS:
        return "Negative"
    return "Neutral"


def clean_text(text: str) -> str:
    """Light cleaning that mirrors the training pipeline."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


# LinearSVC has no predict_proba, so the decision-function margins are turned
# into a probability-like score with a temperature-scaled softmax. A lower
# temperature sharpens the distribution (the model has ~100 output classes).
SOFTMAX_TEMPERATURE = 0.2


def softmax(x: np.ndarray, temperature: float = SOFTMAX_TEMPERATURE) -> np.ndarray:
    x = (x - np.max(x)) / temperature
    e = np.exp(x)
    return e / e.sum()


def predict(text: str):
    """Return (raw_label, polarity, confidence, top3)."""
    vec = vectorizer.transform([clean_text(text)])
    label = model.predict(vec)[0]

    top3, conf, probs = [], None, None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
    elif hasattr(model, "decision_function"):
        scores = np.atleast_2d(model.decision_function(vec))[0]
        if scores.size > 1:
            probs = softmax(scores)
        else:
            conf = float(min(abs(scores[0]) / 2.0, 1.0))

    if probs is not None:
        classes = [str(c).strip() for c in model.classes_]
        order = np.argsort(probs)[::-1][:3]
        top3 = [(classes[i], float(probs[i])) for i in order]
        conf = float(probs.max())

    return str(label).strip(), to_polarity(label), conf, top3


# =========================================================
# 5. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        "<div class='side-title'>🧠 Sentiment Analyser</div>"
        "<div class='side-sub'>TF-IDF · Linear SVM</div>",
        unsafe_allow_html=True,
    )

    page = st.radio("Navigation", ["Analyse", "Dataset", "About"],
                    label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    status = "Model loaded" if MODEL_OK else "Model not loaded"
    st.markdown(
        f"<div class='badge'><span class='dot'></span>{status}</div>"
        "<div class='badge'>⚙️ Vectoriser · TF-IDF</div>"
        "<div class='badge'>📐 Classifier · Linear SVM</div>",
        unsafe_allow_html=True,
    )
    if MODEL_OK:
        st.markdown(f"<div class='badge'>🗂 Dataset · {len(df):,} samples</div>",
                    unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div class='side-sub'>Developed by<br>"
        "<b style='color:#E8ECF3'>Agam Nigam</b></div>",
        unsafe_allow_html=True,
    )

if not MODEL_OK:
    st.error(f"Could not load the model files. Details: {LOAD_ERROR}")
    st.stop()

# =========================================================
# 6. ANALYSE PAGE
# =========================================================
if page == "Analyse":
    st.markdown(
        """
        <div class="hero">
            <h1>Sentiment Analysis Engine</h1>
            <p>Type or paste any sentence, review or social-media post. The model converts it
            into TF-IDF features and predicts the underlying emotion and overall polarity.</p>
            <span class="pill">Python</span>
            <span class="pill">Scikit-learn</span>
            <span class="pill">TF-IDF</span>
            <span class="pill">Linear SVM</span>
            <span class="pill">Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # "text" is a plain state key (NOT a widget key) so it can be written to.
    # "input_nonce" rotates the widget key, which forces the text area to
    # re-initialise whenever we inject a sample or clear the box.
    if "text" not in st.session_state:
        st.session_state.text = ""
    if "input_nonce" not in st.session_state:
        st.session_state.input_nonce = 0

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown("<div class='card'><h3>Input text</h3>", unsafe_allow_html=True)
        user_input = st.text_area(
            "input",
            value=st.session_state.text,
            key=f"text_input_{st.session_state.input_nonce}",
            height=200,
            label_visibility="collapsed",
            placeholder="Example: I really enjoyed this movie — the acting was fantastic!",
        )
        st.session_state.text = user_input
        b1, b2, _ = st.columns([1, 1, 2])
        with b1:
            analyze = st.button("🚀 Analyse", type="primary")
        with b2:
            clear = st.button("🗑 Clear")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>Try a sample</h3>", unsafe_allow_html=True)
        samples = {
            "Positive": "Finished an amazing workout today and I feel absolutely fantastic!",
            "Negative": "Traffic was terrible this morning and it completely ruined my day.",
            "Neutral": "The meeting has been moved to Thursday afternoon.",
        }
        cols = st.columns(3)
        for col, (name, sentence) in zip(cols, samples.items()):
            with col:
                if st.button(name, key=f"sample_{name}"):
                    st.session_state.text = sentence
                    st.session_state.input_nonce += 1
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if clear:
            st.session_state.text = ""
            st.session_state.input_nonce += 1
            st.rerun()

    with right:
        st.markdown("<div class='card'><h3>Result</h3>", unsafe_allow_html=True)

        if analyze and user_input.strip():
            label, polarity, conf, top3 = predict(user_input)
            emoji, css_class = POLARITY_META[polarity]

            st.markdown(
                f"""
                <div class="result {css_class}">
                    <div class="sub">Predicted emotion</div>
                    <div class="label">{emoji} {label.upper()}</div>
                    <div class="sub">Overall polarity · {polarity}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if conf is not None:
                st.write("")
                st.caption(f"Model confidence — {conf * 100:.1f}%")
                st.progress(min(max(conf, 0.0), 1.0))

            if top3:
                st.caption("Closest emotion classes")
                top_df = pd.DataFrame(top3, columns=["Emotion", "Score"])
                top_df.insert(1, "Polarity", top_df["Emotion"].apply(to_polarity))
                st.dataframe(
                    top_df,
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score", min_value=0.0, max_value=1.0, format="%.1f%%"
                        )
                    },
                )

            words = len(user_input.split())
            chars = len(user_input)
            sents = max(1, sum(user_input.count(p) for p in ".!?"))
            m1, m2, m3 = st.columns(3)
            for col, key, val in ((m1, "Words", words),
                                  (m2, "Characters", chars),
                                  (m3, "Sentences", sents)):
                col.markdown(
                    f"<div class='stat'><div class='v'>{val}</div>"
                    f"<div class='k'>{key}</div></div>",
                    unsafe_allow_html=True,
                )
        elif analyze:
            st.warning("Please enter some text first.")
        else:
            st.markdown(
                "<div style='color:#8B95A7;font-size:14px;line-height:1.7'>"
                "Enter text on the left and press <b>Analyse</b>.<br><br>"
                "This panel will show the predicted polarity, the fine-grained emotion "
                "label, a confidence score and basic text statistics.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 7. DATASET PAGE
# =========================================================
elif page == "Dataset":
    st.markdown(
        "<div class='hero'><h1>Dataset Overview</h1>"
        "<p>Cleaned social-media dataset used to train and evaluate the classifier.</p></div>",
        unsafe_allow_html=True,
    )

    polarity_series = df["Sentiment"].apply(to_polarity)
    tiles = [
        ("Samples", f"{len(df):,}"),
        ("Emotion labels", df["Sentiment"].nunique()),
        ("Positive", int((polarity_series == "Positive").sum())),
        ("Negative", int((polarity_series == "Negative").sum())),
    ]
    for col, (k, v) in zip(st.columns(4), tiles):
        col.markdown(
            f"<div class='stat'><div class='v'>{v}</div><div class='k'>{k}</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='card'><h3>Top emotions</h3>", unsafe_allow_html=True)
        st.bar_chart(df["Sentiment"].value_counts().head(10))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>Polarity distribution</h3>", unsafe_allow_html=True)
        st.bar_chart(polarity_series.value_counts())
        st.markdown("</div>", unsafe_allow_html=True)

    if "Platform" in df.columns:
        st.markdown("<div class='card'><h3>Posts per platform</h3>", unsafe_allow_html=True)
        st.bar_chart(df["Platform"].value_counts())
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><h3>Sample records</h3>", unsafe_allow_html=True)
    show_cols = [c for c in ["Text", "Clean_Text", "Sentiment", "Platform", "Country"]
                 if c in df.columns]
    st.dataframe(df[show_cols].head(25), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 8. ABOUT PAGE
# =========================================================
else:
    st.markdown(
        "<div class='hero'><h1>About the Project</h1>"
        "<p>A machine-learning pipeline that classifies short text into emotional "
        "categories and an overall positive / negative / neutral polarity.</p></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            "<div class='card'><h3>Pipeline</h3>"
            "<div style='line-height:2;font-size:14.5px'>"
            "1. Text cleaning — lowercasing, URL / mention removal, punctuation stripping<br>"
            "2. Feature extraction — TF-IDF vectorisation<br>"
            "3. Classification — Linear Support Vector Machine<br>"
            "4. Post-processing — emotion label mapped to polarity<br>"
            "5. Deployment — Streamlit web interface"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='card'><h3>Tech stack</h3>"
            "<div style='line-height:2;font-size:14.5px'>"
            "Python 3 · Pandas · NumPy · Scikit-learn · Joblib · Streamlit"
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='card'><h3>Features</h3>"
        "<div style='line-height:2;font-size:14.5px'>"
        "• Real-time sentiment prediction<br>"
        "• Fine-grained emotion label with confidence score<br>"
        "• Top-3 candidate emotions<br>"
        "• Dataset explorer with distribution charts<br>"
        "• Text statistics (words, characters, sentences)"
        "</div></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='foot'>Built with Python · Scikit-learn · Streamlit &nbsp;|&nbsp; "
    "Sentiment Analysis using Machine Learning</div>",
    unsafe_allow_html=True,
)
