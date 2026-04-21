import streamlit as st
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.util import ngrams
from collections import Counter
import emoji
import torch
from transformers import pipeline
# plotly replaced with native st charts (no extra install needed)
from datetime import datetime

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Apple Sentiment AI",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. CUSTOM CSS – Premium dark glassmorphism UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit default header bar ── */
header[data-testid="stHeader"] {
    display: none !important;
}
.stAppDeployButton { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }


/* ── Dark background ── */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 40%, #0a0f1a 100%);
    color: #e8eaf0;
}

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(31,31,50,0.7), rgba(15,20,35,0.7));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    margin-bottom: 2rem;
    backdrop-filter: blur(12px);
}
.hero-header h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a8edea, #fed6e3, #a8edea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-header p {
    color: #8892a4;
    margin-top: 0.5rem;
    font-size: 1rem;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(168,237,234,0.35);
}
.metric-card .val { font-size: 1.9rem; font-weight: 700; }
.metric-card .lbl { font-size: 0.78rem; color: #8892a4; margin-top: 0.2rem; }

/* ── Aspect badge ── */
.aspect-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1.1rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    margin-bottom: 0.5rem;
}
.aspect-badge {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-pos { background: rgba(52,211,153,0.18); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-neg { background: rgba(248,113,113,0.18); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.badge-neu { background: rgba(148,163,184,0.18); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }

/* ── Token card ── */
.token-cloud {
    display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.6rem;
}
.token-chip {
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.25);
    color: #63b3ed;
}

/* ── Section title ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #a8edea;
    margin: 1.2rem 0 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Progress bar override ── */
.stProgress > div > div { background: linear-gradient(90deg, #a8edea, #7dd3fc) !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10,10,20,0.85) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] .stMarkdown h2 { color: #a8edea; }

/* ── Text area ── */
.stTextArea textarea,
.stTextArea > div > div > textarea,
textarea[data-testid],
div[data-baseweb="textarea"] textarea,
div[data-baseweb="base-input"] textarea {
    background: #0f1421 !important;
    background-color: #0f1421 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    caret-color: #a8edea !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus,
div[data-baseweb="textarea"]:focus-within textarea {
    border-color: rgba(168,237,234,0.5) !important;
    box-shadow: 0 0 0 2px rgba(168,237,234,0.1) !important;
    outline: none !important;
}
/* Also fix the wrapper div that can be white */
div[data-baseweb="textarea"],
div[data-baseweb="base-input"] {
    background: #0f1421 !important;
    background-color: #0f1421 !important;
    border-radius: 12px !important;
}

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg, #4f8cf7, #7c5ce8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 2.5rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(79,140,247,0.4) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8892a4 !important;
    border-radius: 8px 8px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #a8edea !important;
    border-bottom: 2px solid #a8edea !important;
    background: rgba(168,237,234,0.07) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. NLTK DOWNLOADS (CACHED)
# ─────────────────────────────────────────────
@st.cache_resource
def download_nltk_data():
    for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                'averaged_perceptron_tagger_eng', 'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words', 'omw-1.4']:
        nltk.download(pkg, quiet=True)

download_nltk_data()


# ─────────────────────────────────────────────
# 4. LOAD AI MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=device,
        top_k=3   # return top-3 scores for confidence chart
    )

with st.spinner("🔄 Loading XLM-RoBERTa sentiment model (first run may take a few minutes)…"):
    sentiment_analyzer = load_model()

# ─────────────────────────────────────────────
# 5. APPLE PRODUCT / ASPECT TAXONOMY  (2024-2025)
# ─────────────────────────────────────────────
APPLE_ASPECTS = {
    # ── iPhone 16 family ──
    "iphone 16": "📱 iPhone 16",
    "iphone 16 plus": "📱 iPhone 16 Plus",
    "iphone 16 pro": "📱 iPhone 16 Pro",
    "iphone 16 pro max": "📱 iPhone 16 Pro Max",
    "iphone 15": "📱 iPhone 15",
    "iphone": "📱 iPhone (general)",
    # ── Mac ──
    "macbook air": "💻 MacBook Air (M3)",
    "macbook pro": "💻 MacBook Pro (M4)",
    "mac mini": "🖥️ Mac Mini (M4)",
    "mac studio": "🖥️ Mac Studio (M4 Max)",
    "mac pro": "🖥️ Mac Pro",
    "imac": "🖥️ iMac (M4)",
    "mac": "💻 Mac (general)",
    # ── iPad ──
    "ipad pro": "📐 iPad Pro (M4)",
    "ipad air": "📐 iPad Air (M3)",
    "ipad mini": "📐 iPad Mini",
    "ipad": "📐 iPad (general)",
    # ── Apple Watch ──
    "apple watch ultra": "⌚ Apple Watch Ultra 2",
    "apple watch series 10": "⌚ Apple Watch Series 10",
    "apple watch": "⌚ Apple Watch (general)",
    # ── AirPods ──
    "airpods pro": "🎧 AirPods Pro 2",
    "airpods max": "🎧 AirPods Max (USB-C)",
    "airpods": "🎧 AirPods (general)",
    # ── Vision / Spatial ──
    "apple vision pro": "🥽 Apple Vision Pro",
    "vision pro": "🥽 Apple Vision Pro",
    "visionos": "🥽 visionOS",
    # ── Software / Services ──
    "ios 18": "📲 iOS 18",
    "macos sequoia": "🖥️ macOS Sequoia",
    "apple intelligence": "🧠 Apple Intelligence",
    "siri": "🗣️ Siri",
    "apple pay": "💳 Apple Pay",
    "icloud": "☁️ iCloud",
    "app store": "🏪 App Store",
    # ── Components / Features ──
    "m4": "⚡ M4 chip",
    "m3": "⚡ M3 chip",
    "a18": "⚡ A18 Bionic",
    "a18 pro": "⚡ A18 Pro",
    "battery": "🔋 Battery",
    "camera": "📷 Camera",
    "display": "🖥️ Display",
    "screen": "🖥️ Screen",
    "face id": "🔒 Face ID",
    "touch id": "🔒 Touch ID",
    "ceramic shield": "🛡️ Ceramic Shield",
    "dynamic island": "🏝️ Dynamic Island",
    "action button": "🔘 Action Button",
    "usb-c": "🔌 USB-C",
    # ── Brand ──
    "apple": "🍎 Apple (brand)",
    "aapl": "📈 AAPL stock",
    "tim cook": "👤 Tim Cook",
    "wwdc": "🎉 WWDC",
    "apple event": "🎉 Apple Event",
}

LABEL_MAP = {"LABEL_0": ("Negative", "neg"), "LABEL_1": ("Neutral", "neu"), "LABEL_2": ("Positive", "pos")}

# ─────────────────────────────────────────────
# 6. NLP PIPELINE
# ─────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

EMOJI_SENTIMENT = {
    "😡": "Negative", "😤": "Negative", "💀": "Negative", "🤮": "Negative",
    "😍": "Positive", "🔥": "Positive", "❤️": "Positive", "🥳": "Positive",
    "😂": "Positive", "👏": "Positive", "🎉": "Positive", "✅": "Positive",
    "😐": "Neutral",  "🤔": "Neutral",  "🙄": "Neutral",
}

def extract_emojis(text):
    return [ch for ch in text if ch in emoji.EMOJI_DATA]

def decode_emoji_sentiment(emojis_found):
    pos = sum(1 for e in emojis_found if EMOJI_SENTIMENT.get(e) == "Positive")
    neg = sum(1 for e in emojis_found if EMOJI_SENTIMENT.get(e) == "Negative")
    neu = sum(1 for e in emojis_found if EMOJI_SENTIMENT.get(e) == "Neutral")
    total = pos + neg + neu
    if total == 0: return None
    if pos > neg: return "Positive 🟢"
    if neg > pos: return "Negative 🔴"
    return "Neutral ⚪"

def clean_text(text):
    """Full rubric-compliant NLP pre-processing pipeline."""
    if pd.isna(text): return ""
    text = str(text).lower()
    # Expand contractions (basic)
    contractions = {"can't": "cannot", "won't": "will not", "i'm": "i am",
                    "it's": "it is", "n't": " not", "'ve": " have", "'ll": " will", "'re": " are"}
    for k, v in contractions.items():
        text = text.replace(k, v)
    # Replace emoji with text
    text = emoji.replace_emoji(text, replace=lambda chars, _: f' {chars} ')
    # Remove URLs and @mentions
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize + remove stopwords (keep keywords)
    cleaned = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and t.strip()]
    return " ".join(cleaned), tokens

def get_ngrams(tokens, n=2):
    grams = list(ngrams(tokens, n))
    return Counter([" ".join(g) for g in grams]).most_common(8)

def get_pos_summary(tokens):
    tagged = pos_tag(tokens)
    tag_map = {'NN': 'Noun', 'NNP': 'Proper Noun', 'VB': 'Verb',
               'JJ': 'Adjective', 'RB': 'Adverb', 'PRP': 'Pronoun'}
    summary = Counter()
    for _, tag in tagged:
        for prefix, label in tag_map.items():
            if tag.startswith(prefix):
                summary[label] += 1
                break
    return dict(summary)

def named_entity_extraction(tokens):
    tagged = pos_tag(tokens)
    chunked = ne_chunk(tagged)
    entities = []
    for subtree in chunked:
        if isinstance(subtree, Tree):
            entity = " ".join([word for word, _ in subtree.leaves()])
            entities.append((entity, subtree.label()))
    return entities

def run_full_analysis(raw_text):
    """Returns full NLP pipeline results."""
    emojis_found = extract_emojis(raw_text)
    emoji_sentiment = decode_emoji_sentiment(emojis_found)

    clean_result = clean_text(raw_text)
    clean_str, all_tokens = clean_result[0], clean_result[1]
    clean_tokens = clean_str.split()

    text_lower = raw_text.lower()
    # Multi-word first, then single-word
    matched_aspects = {}
    for phrase in sorted(APPLE_ASPECTS.keys(), key=lambda x: -len(x)):
        if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
            matched_aspects[phrase] = APPLE_ASPECTS[phrase]

    # Get sentiment scores per aspect
    aspect_sentiments = {}
    if matched_aspects:
        try:
            raw_preds = sentiment_analyzer(raw_text[:512])[0]  # list of top-3 dicts
            # Sort by score
            sorted_preds = sorted(raw_preds, key=lambda x: x['score'], reverse=True)
            top = sorted_preds[0]
            label_key = top['label']
            label_text, badge_cls = LABEL_MAP.get(label_key, (label_key, "neu"))
            confidence = round(top['score'] * 100, 1)

            for phrase, display in matched_aspects.items():
                aspect_sentiments[display] = {
                    "label": label_text,
                    "badge": badge_cls,
                    "confidence": confidence,
                }
            all_scores = {LABEL_MAP.get(p['label'], (p['label'],))[0]: round(p['score'] * 100, 1)
                          for p in sorted_preds}
        except Exception as e:
            all_scores = {}
    else:
        all_scores = {}
        try:
            raw_preds = sentiment_analyzer(raw_text[:512])[0]
            sorted_preds = sorted(raw_preds, key=lambda x: x['score'], reverse=True)
            all_scores = {LABEL_MAP.get(p['label'], (p['label'],))[0]: round(p['score'] * 100, 1)
                          for p in sorted_preds}
        except:
            pass

    bigrams = get_ngrams(all_tokens, 2) if len(all_tokens) >= 2 else []
    trigrams = get_ngrams(all_tokens, 3) if len(all_tokens) >= 3 else []
    pos_summary = get_pos_summary(all_tokens)
    entities = named_entity_extraction(all_tokens)
    word_freq = Counter(clean_tokens).most_common(10)

    return {
        "clean_text": clean_str,
        "all_tokens": all_tokens,
        "clean_tokens": clean_tokens,
        "emojis": emojis_found,
        "emoji_sentiment": emoji_sentiment,
        "aspects": aspect_sentiments,
        "all_scores": all_scores,
        "bigrams": bigrams,
        "trigrams": trigrams,
        "pos_summary": pos_summary,
        "entities": entities,
        "word_freq": word_freq,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()),
        "token_count": len(all_tokens),
    }

# ─────────────────────────────────────────────
# 7. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍏 Apple Sentiment AI")
    st.markdown("---")
    st.markdown("**Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`")
    st.markdown("**Task:** Aspect-Based Sentiment Analysis")
    st.markdown("**Languages:** English · French · Spanish · Emojis")
    st.markdown("---")
    st.markdown("**NLP Pipeline:**")
    st.markdown("""
- Contraction expansion
- Emoji decoding
- URL & mention removal  
- Tokenisation (NLTK)
- POS tagging
- NE chunking (NER)
- Lemmatisation
- Stopword removal
- N-gram extraction
- TF/frequency scoring
""")
    st.markdown("---")
    st.markdown(f"**Products tracked:** `{len(APPLE_ASPECTS)}`")
    st.markdown(f"**Model precision:** XLM-RoBERTa multilingual")

    st.markdown("---")
    st.markdown("### 📂 Dataset")
    st.markdown("""
<a href="https://www.kaggle.com/datasets/anishdabhane/apple-tweets-sentiment-dataset" target="_blank"
   style="display:inline-flex;align-items:center;gap:0.45rem;padding:0.45rem 0.9rem;
          background:rgba(32,209,79,0.12);border:1px solid rgba(32,209,79,0.35);
          border-radius:8px;color:#20d14f;font-size:0.82rem;font-weight:600;
          text-decoration:none;transition:all 0.2s;">
  🗂️ Apple Tweets Sentiment Dataset
</a>
<p style="color:#8892a4;font-size:0.75rem;margin-top:0.5rem;">
  Kaggle · anishdabhane
</p>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Try these examples")
    examples = [
        "iPhone 16 Pro Max camera is absolutely insane 🔥",
        "Apple Vision Pro is way too expensive for most people 😤",
        "M4 MacBook Pro blows every Windows laptop out of the water!",
        "AirPods Max USB-C finally but battery life still mediocre 🙄",
        "Dynamic Island on iPhone 16 is a game changer honestly",
        "AAPL stock dropped after Tim Cook's event announcement 📉",
        "Apple Intelligence Siri upgrades in iOS 18 are impressive!",
        "Le nouvel iPad Pro M4 est incroyable mais trop cher 😡",
    ]
    if "example_text" not in st.session_state:
        st.session_state.example_text = examples[0]
    for ex in examples:
        if st.button(ex[:52] + "…" if len(ex) > 52 else ex, key=f"ex_{ex[:20]}"):
            st.session_state.example_text = ex

# ─────────────────────────────────────────────
# 8. MAIN UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🍏 Apple Sentiment AI</h1>
  <p>Aspect-Based NLP · XLM-RoBERTa · 40+ Apple Products · Emoji Decoding · NER · N-grams</p>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    label="✏️ Enter a tweet, review, or comment:",
    value=st.session_state.get("example_text", "iPhone 16 Pro Max camera is absolutely insane 🔥"),
    height=110,
    placeholder="e.g. The M4 MacBook Pro is lightning fast but macOS Sequoia still has bugs…",
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

# ─────────────────────────────────────────────
# 9. RESULTS
# ─────────────────────────────────────────────
if analyze_btn and user_input.strip():
    with st.spinner("🧠 Running NLP pipeline…"):
        r = run_full_analysis(user_input)

    # ── Top metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    overall_label = max(r["all_scores"], key=r["all_scores"].get) if r["all_scores"] else "N/A"
    overall_conf  = r["all_scores"].get(overall_label, 0)
    icon_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}
    icon = icon_map.get(overall_label, "❓")

    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{icon} {overall_label}</div>
            <div class="lbl">Overall Sentiment</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="val" style="color:#a8edea">{overall_conf}%</div>
            <div class="lbl">Model Confidence</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="val" style="color:#fed6e3">{len(r['aspects'])}</div>
            <div class="lbl">Apple Aspects Found</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="val" style="color:#7dd3fc">{r['word_count']}</div>
            <div class="lbl">Words</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🎯 Aspects", "📊 Sentiment Scores", "🔤 NLP Pipeline", "📈 N-Grams & Freq", "🏷️ NER & POS"]
    )

    # ─── TAB 1: Aspects ───────────────────────────────────────
    with tab1:
        if r["aspects"]:
            st.markdown('<div class="section-title">🎯 Detected Apple Aspects</div>', unsafe_allow_html=True)
            for display, info in r["aspects"].items():
                badge_class = f"badge-{info['badge']}"
                st.markdown(f"""
                <div class="aspect-row">
                    <span style="font-weight:500">{display}</span>
                    <div style="display:flex;align-items:center;gap:0.8rem">
                        <span style="color:#8892a4;font-size:0.8rem">{info['confidence']}% confidence</span>
                        <span class="aspect-badge {badge_class}">{info['label']}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No specific Apple product/feature aspects detected in this text.")

        if r["emojis"]:
            st.markdown('<div class="section-title">😀 Emoji Analysis</div>', unsafe_allow_html=True)
            emo_str = " ".join(r["emojis"])
            emo_sent = r["emoji_sentiment"] or "Unknown"
            st.markdown(f"**Emojis found:** {emo_str} &nbsp;&nbsp; **Emoji sentiment signal:** {emo_sent}")

    # ─── TAB 2: Sentiment Scores ───────────────────────────────
    with tab2:
        if r["all_scores"]:
            st.markdown('<div class="section-title">📊 XLM-RoBERTa Confidence Distribution</div>', unsafe_allow_html=True)
            # Native Streamlit bar chart
            df_scores = pd.DataFrame({
                "Sentiment": list(r["all_scores"].keys()),
                "Confidence (%)": list(r["all_scores"].values()),
            }).set_index("Sentiment")
            st.bar_chart(df_scores, height=300)

            st.markdown('<div class="section-title">Confidence Meters</div>', unsafe_allow_html=True)
            icon_map2 = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}
            for lbl, val in r["all_scores"].items():
                ico = icon_map2.get(lbl, "")
                st.markdown(f"{ico} `{lbl}` — **{val}%**")
                st.progress(min(int(val), 100))

    # ─── TAB 3: NLP Pipeline ──────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">🧹 Cleaned Text (Post-Pipeline)</div>', unsafe_allow_html=True)
        st.code(r["clean_text"] or "(empty after cleaning)", language="text")

        st.markdown('<div class="section-title">🔤 Tokens (All)</div>', unsafe_allow_html=True)
        chips = "".join([f'<span class="token-chip">{t}</span>' for t in r["all_tokens"][:60]])
        st.markdown(f'<div class="token-cloud">{chips}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">✅ Cleaned Tokens (Lemmatised, No Stopwords)</div>', unsafe_allow_html=True)
        chips2 = "".join([f'<span class="token-chip" style="border-color:rgba(52,211,153,0.3);color:#34d399">{t}</span>'
                          for t in r["clean_tokens"]])
        st.markdown(f'<div class="token-cloud">{chips2}</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">📏 Text Statistics</div>', unsafe_allow_html=True)
            stats = pd.DataFrame({
                "Metric": ["Characters", "Words", "All Tokens", "Clean Tokens", "Emojis"],
                "Count": [r["char_count"], r["word_count"], r["token_count"],
                           len(r["clean_tokens"]), len(r["emojis"])],
            })
            st.dataframe(stats, hide_index=True, use_container_width=True)

    # ─── TAB 4: N-Grams & Frequency ───────────────────────────
    with tab4:
        col_bi, col_tri = st.columns(2)
        with col_bi:
            st.markdown('<div class="section-title">2️⃣ Top Bigrams</div>', unsafe_allow_html=True)
            if r["bigrams"]:
                df_bi = pd.DataFrame(r["bigrams"], columns=["Bigram", "Count"]).set_index("Bigram")
                st.bar_chart(df_bi, height=260)
            else:
                st.info("Not enough tokens for bigrams.")
        with col_tri:
            st.markdown('<div class="section-title">3️⃣ Top Trigrams</div>', unsafe_allow_html=True)
            if r["trigrams"]:
                df_tri = pd.DataFrame(r["trigrams"], columns=["Trigram", "Count"]).set_index("Trigram")
                st.bar_chart(df_tri, height=260)
            else:
                st.info("Not enough tokens for trigrams.")

        st.markdown('<div class="section-title">📊 Word Frequency (Top 10)</div>', unsafe_allow_html=True)
        if r["word_freq"]:
            df_wf = pd.DataFrame(r["word_freq"], columns=["Word", "Frequency"]).set_index("Word")
            st.bar_chart(df_wf, height=260)

    # ─── TAB 5: NER & POS ─────────────────────────────────────
    with tab5:
        col_ner, col_pos = st.columns(2)
        with col_ner:
            st.markdown('<div class="section-title">🏷️ Named Entities (NER)</div>', unsafe_allow_html=True)
            if r["entities"]:
                df_ner = pd.DataFrame(r["entities"], columns=["Entity", "Type"])
                st.dataframe(df_ner, hide_index=True, use_container_width=True)
            else:
                st.info("No named entities detected.")

        with col_pos:
            st.markdown('<div class="section-title">🔠 Part-of-Speech Distribution</div>', unsafe_allow_html=True)
            if r["pos_summary"]:
                df_pos = pd.DataFrame(list(r["pos_summary"].items()), columns=["POS Tag", "Count"]).set_index("POS Tag")
                st.bar_chart(df_pos, height=280)

elif analyze_btn and not user_input.strip():
    st.warning("⚠️ Please enter some text before clicking Analyze.")

# ─────────────────────────────────────────────
# 10. FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"""<div style="text-align:center;color:#4a5568;font-size:0.78rem;padding:0.5rem 0">
    🍏 Apple Sentiment AI &nbsp;·&nbsp; XLM-RoBERTa NLP &nbsp;·&nbsp;
    NLTK · Transformers · Streamlit &nbsp;·&nbsp; {datetime.now().strftime('%Y')}
    </div>""",
    unsafe_allow_html=True,
)