"""
thematic_analysis.py
=====================
Computational Thematic Analysis Pipeline
Dissertation: Beyond the Eval — Philippine HEI Subreddit Study

Runs three complementary algorithms on the cleaned comment corpus:
  1. TF-IDF + NMF        — fast, interpretable, produces 10 named topics
  2. LDA (Gensim)        — probabilistic generative model, topic coherence scored
  3. Keyword extraction  — RAKE algorithm for multi-word phrases per theme

Each method produces:
  • Per-comment topic assignments
  • Per-topic keyword lists
  • Per-subreddit topic distributions
  • Tie-type × topic cross-table (Granovetter lens)
  • Temporal topic trends
  • NVivo-ready export CSV

Then maps computational topics to the four qualitative codebook domains:
  A: Faculty Performance Evaluation
  B: Institutional and Course Evaluation
  C: Evaluative Rhetoric
  D: Community Norms and Filipino Cultural Frames

Outputs
-------
Written next to the source CSV, under a timestamped run folder:

  {DATA_DIR}/thematic_analysis/{timestamp}/outputs/
    ├── k_comparison.csv                 — LDA coherence + unclassified % per k
    ├── k_comparison_coherence.png       — coherence-vs-k chart (pick best k)
    ├── k10/
    │     ├── theme_summary.csv          — per-topic keywords + codebook alignment
    │     ├── topic_codebook_alignment.csv — transparent per-topic alignment audit
    │     ├── codebook_coverage.csv      — how emergent topics map onto the codebook
    │     ├── inductive_candidates.csv   — unaligned/ambiguous topics + exemplars
    │     ├── comments_with_topics.csv   — full corpus with topic assignments
    │     ├── topic_by_subreddit.csv     — topic distribution per HEI
    │     ├── topic_by_tie_type.csv      — topic × tie type (Granovetter)
    │     ├── topic_by_month.csv         — temporal trends
    │     ├── rake_keywords_by_domain.csv
    │     ├── nvivo_thematic_export.csv  — NVivo-ready (blank coding columns)
    │     ├── figures/theme_*.png        — 6 visualization figures
    │     └── lda_model/                 — saved LDA model for inspection
    ├── k15/  (same structure)
    └── k20/  (same structure)

Requirements
------------
  pip install scikit-learn gensim nltk matplotlib pandas numpy

Usage
-----
  python thematic_analysis.py                 # runs k = 10, 15, 20

  # Quick run on a subset (faster, for testing)
  python thematic_analysis.py --sample 5000

  # Custom set of k values
  python thematic_analysis.py --k-values 8,10,12,15

  # Only regenerate figures from an existing run
  python thematic_analysis.py --only-figures \
      --run-dir "/content/drive/.../thematic_analysis/20260529_104500/outputs"
"""

import os
import re
import sys
import math
import argparse
import warnings
import logging
import pickle
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── NLTK data ─────────────────────────────────────────────────────────────────
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ── Configuration ─────────────────────────────────────────────────────────────
# Directory that holds the data. load_data() prefers the canonical cleaned file
# from Module 1 at {DATA_DIR}/cleaned/comments_clean.csv (so cleaning and tie_type
# match the sentiment and network strands), and falls back to the raw
# {DATA_DIR}/dataset_comments.csv only if the cleaned file is absent.
DATA_DIR   = "/content/drive/MyDrive/2026 Dissertation/dissertation dataset"

# All output for this run goes under:
#   {DATA_DIR}/thematic_analysis/{timestamp}/outputs/k{K}/
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = os.path.join(DATA_DIR, "thematic_analysis", TIMESTAMP, "outputs")

# Topic counts (k) to run. Each gets its own k{K}/ subfolder.
K_VALUES   = [10, 15, 20]

# OUT_DIR / FIG_DIR / MODEL_DIR are reassigned per-k inside run_for_k().
OUT_DIR    = None
FIG_DIR    = None
MODEL_DIR  = None

N_TOPICS   = 10         # default used by run_nmf/run_lda signatures only
MIN_WORDS  = 8          # exclude very short comments
MAX_VOCAB  = 5000       # TF-IDF/CountVec vocabulary ceiling
LDA_PASSES = 15         # more passes = better coherence, slower
LDA_ITER   = 400
RANDOM_SEED = 42

# ── Color palette (per HEI + tie type) ───────────────────────────────────────
SUB_C = {
    "Benilde": "#88E788",
    "dlsu":    "#198754",
    "peyups":  "#dc3545",
    "AdMU":    "#0d6efd",
    "unknown": "#adb5bd",
}
TIE_C = {"weak": "#fd7e14", "strong": "#4dabf7"}
BG, SF, TX, MU, GR = "#FFFFFF", "#F8F9FA", "#212529", "#6C757D", "#DEE2E6"
SAVE = dict(dpi=180, bbox_inches="tight", facecolor=BG, edgecolor="none")

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SF, "axes.edgecolor": GR,
    "text.color": TX, "axes.labelcolor": TX,
    "xtick.color": MU, "ytick.color": MU,
    "grid.color": GR, "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Bot list ──────────────────────────────────────────────────────────────────
# Mirrors 01_load_and_clean.py (Module 1) exactly. Only used on the raw-file
# fallback path; when the canonical cleaned/comments_clean.csv is read, cleaning
# has already been applied upstream and this list is not consulted.
BOTS = {
    "AutoModerator", "BotDefense", "RepostSleuthBot", "RemindMeBot",
    "MAGIC_EYE_BOT", "anti-gif-bot", "Bot_Metric", "reddit-bot",
    "Deleted", "deleted", "[deleted]",
}

# ── Filipino + English stopwords combined ─────────────────────────────────────
FIL_STOPS = {
    "ang", "ng", "na", "sa", "at", "ay", "ni", "mga", "kung", "ito",
    "ko", "mo", "ka", "po", "din", "rin", "nga", "ba", "si", "naman",
    "yung", "yun", "ung", "yon", "siya", "niya", "nila", "kami", "tayo",
    "kayo", "sila", "ako", "ikaw", "kasi", "pero", "lang", "talaga",
    "daw", "raw", "pala", "natin", "namin", "ninyo", "wala", "may",
    "mayroon", "oo", "opo", "hindi", "huwag", "para", "saan", "paano",
    "bakit", "kailan", "sino", "ano", "eh", "kaya", "pag", "kapag",
    "kahit", "habang", "bago", "pagkatapos", "ha", "haha", "hehe",
    "lol", "lmao", "hmm", "hm", "ah", "oh", "yep", "yeah", "ok",
    "okay", "im", "ive", "id", "dont", "doesnt", "didnt", "isnt",
    "wasnt", "arent", "weren", "ive", "weve", "theyre", "theyll",
    "its", "thats", "isnt", "arent", "wasnt", "shouldnt", "couldnt",
    "wouldnt", "nd", "ta", "di", "mas", "rin",
}
EN_STOPS = set(stopwords.words("english"))
ALL_STOPS = FIL_STOPS | EN_STOPS | {
    # Reddit-specific noise
    "deleted", "removed", "edit", "reddit", "post", "thread", "comment",
    "subreddit", "upvote", "downvote", "mod", "moderator", "bot",
    "http", "https", "www", "com", "link", "url", "image", "gif",
    "crosspost", "repost", "oc", "tldr", "tl", "dr", "edt",
    # Generic filler
    "really", "just", "also", "even", "would", "could", "should",
    "will", "can", "get", "got", "going", "said", "know", "think",
    "much", "still", "well", "back", "around", "though", "already",
    "actually", "probably", "maybe", "something", "someone", "people",
    "thing", "things", "way", "ways", "lot", "lots", "bit", "bit",
    "thank", "thanks", "please", "ask", "sure", "try", "need", "want",
    "hope", "say", "tell", "make", "made", "look", "looks", "feel",
}


# ── Preprocessing ─────────────────────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Normalize Reddit comment text for tokenization."""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove Reddit mention patterns (u/user, r/sub)
    text = re.sub(r"\bu/\w+", " ", text)
    text = re.sub(r"\br/\w+", " ", text)
    # Remove special characters but keep apostrophes for contractions
    text = re.sub(r"[^a-z\s']", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize, lemmatize, and remove stopwords."""
    tokens = clean_text(text).split()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha()
        and len(t) > 2
        and t not in ALL_STOPS
    ]
    return tokens


# ── Codebook (a priori template — used for POST-HOC alignment only) ───────────
# These seeds do NOT discover topics. NMF/LDA run fully unsupervised; the codebook
# is applied AFTER discovery to measure how each emergent topic aligns with the
# four-domain qualitative codebook. This implements the hybrid inductive-deductive
# design (Fereday & Muir-Cochrane, 2006; a priori template, Crabtree & Miller,
# 1999): alignment is REPORTED, never forced. Topics with no codebook match are
# surfaced as inductive candidates for the human coding pass, not relabelled.
#
# Rhetorical / epistemic subthemes (C1 evidence type, C2 rhetorical strategy,
# C3 specificity, D1 credibility norms) describe HOW evaluation is expressed, not
# which content words appear, so they are deliberately left unseeded (lexical=False).
# A bag-of-words model cannot recover them; their absence from computational
# alignment is by design and must NOT be read as a finding — they are carried by
# the human coding (see RQ2 protocol §4).

DOMAIN_LABELS = {
    "A": "Faculty Performance Evaluation",
    "B": "Institutional and Course Evaluation",
    "C": "Evaluative Rhetoric and Epistemic Framing",
    "D": "Community Norms and Relational Dynamics",
    "?": "Inductive candidate — no codebook match",
}

# code -> (label, domain, lexical?, seed terms)
CODEBOOK = {
    "A1": ("Teaching effectiveness", "A", True, [
        "teach", "teaching", "explain", "explains", "explained", "lecture",
        "lessons", "clear", "understand", "magaling", "galing", "knowledgeable",
        "engaging", "boring", "prepared", "learn", "learned", "discussion",
        "approachable"]),
    "A2": ("Grading and assessment", "A", True, [
        "grade", "grades", "grading", "exam", "exams", "quiz", "recit",
        "recitation", "rubric", "score", "fail", "failed", "passing", "output",
        "requirements", "curve", "bell", "lenient", "deadline", "points"]),
    "A3": ("Professional conduct", "A", True, [
        "late", "absent", "cancel", "respond", "reply", "email", "bias",
        "biased", "favoritism", "pabor", "terror", "strict", "rude",
        "respectful", "unfair", "fair", "unprofessional", "ghosting",
        "accommodating", "attitude"]),
    "A4": ("Recommendation/avoidance", "A", True, [
        "avoid", "recommend", "beware", "warning", "warn", "suggest", "advice",
        "choose", "pick", "iwasan", "kunin", "drop", "enroll"]),
    "B1": ("Course evaluation", "B", True, [
        "course", "subject", "subjects", "syllabus", "units", "load", "elective",
        "curriculum", "workload", "materials", "modules", "trimester",
        "semester", "prerequisite"]),
    "B2": ("Institutional policies", "B", True, [
        "tuition", "fees", "fee", "enrollment", "registrar", "admin",
        "administration", "policy", "policies", "registration", "system",
        "office", "scholarship", "financial"]),
    "B3": ("Comparative institutional evaluation", "B", True, [
        "dlsu", "ateneo", "admu", "benilde", "peyups", "lasalle", "salle",
        "ust", "mapua", "compared", "comparison", "versus", "kesa", "better"]),
    "C1": ("Evidence type", "C", False, []),
    "C2": ("Rhetorical strategy", "C", False, []),
    "C3": ("Evaluation specificity", "C", False, []),
    "C4": ("Platform reflexivity", "C", True, [
        "reddit", "subreddit", "sub", "thread", "threads", "upvote", "downvote",
        "karma", "repost", "mods", "mod"]),
    "D1": ("Evaluation credibility norms", "D", False, []),
    "D2": ("Solidarity and support", "D", True, [
        "support", "agree", "same", "true", "valid", "sorry", "kaya",
        "kakayanin", "laban", "congrats", "ingat", "fellow", "classmate",
        "batchmate", "sana", "goodluck"]),
    "D3": ("Gatekeeping and exclusion", "D", True, [
        "gatekeep", "elitist", "deserve", "entitled", "exclusive", "poser"]),
    "D4": ("Filipino cultural evaluation frames", "D", True, [
        "terror", "pabor", "bahala", "kulit", "diskarte", "sir", "maam", "ate",
        "kuya", "suki", "hiya", "utang", "padrino", "lusot", "palakasan",
        "iskolar", "mentor"]),
}

# Derived lookups
SEEDSETS = {c: set(seeds) for c, (lab, dom, lex, seeds) in CODEBOOK.items() if lex}
LEXICAL_SUBTHEMES    = [c for c, (lab, dom, lex, seeds) in CODEBOOK.items() if lex]
HUMAN_ONLY_SUBTHEMES = [c for c, (lab, dom, lex, seeds) in CODEBOOK.items() if not lex]

# Alignment thresholds (tunable)
MIN_ALIGN_OVERLAP = 2   # >=2 matched seed terms needed for a confident alignment
AMBIG_MARGIN      = 1    # best must beat a cross-domain runner-up by this margin


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════
def load_data(sample_n: int | None = None) -> pd.DataFrame:
    """
    Load the comment corpus, preferring the canonical cleaned file produced by
    Module 1 (cleaned/comments_clean.csv) so that cleaning rules and the weak/
    strong tie_type are IDENTICAL to the sentiment and network strands.

    If the cleaned file is present, its existing tie_type column is reused as-is
    (single source of truth). Only when an uncleaned raw file is used as a
    fallback are bot/deleted filtering and the tie_type median split applied
    here, using the same bot list and rule as Module 1.

    The thematic-only steps (≥MIN_WORDS filter, and the tokenization in
    preprocess()) are applied on top regardless of source.
    """
    print("Loading data...")
    cleaned_path = os.path.join(DATA_DIR, "cleaned", "comments_clean.csv")
    raw_path     = os.path.join(DATA_DIR, "dataset_comments.csv")

    if os.path.exists(cleaned_path):
        comments_path, from_cleaned = cleaned_path, True
    elif os.path.exists(raw_path):
        comments_path, from_cleaned = raw_path, False
    else:
        raise FileNotFoundError(
            f"No input found. Expected {cleaned_path} (preferred) "
            f"or {raw_path} (raw fallback)."
        )

    c = pd.read_csv(comments_path, parse_dates=["comment_created_utc"])
    print(f"  Source: {comments_path}")

    if not from_cleaned:
        # Raw fallback only — replicate Module 1 cleaning so the corpus matches
        # the other strands. (Skipped when the cleaned file is used.)
        c = (c[~c["comment_author"].isin(BOTS) &
               ~c["comment_body"].isin(["[deleted]", "[removed]"])]
             .dropna(subset=["comment_body"])
             .drop_duplicates("comment_id")
             .copy())
    else:
        c = c.copy()

    # Tie type: reuse the canonical column if present; derive only if missing.
    if "tie_type" in c.columns and c["tie_type"].notna().any():
        print("  Using existing canonical tie_type (from Module 1).")
    else:
        med = c["tie_strength_proxy"].median()
        c["tie_type"] = c["tie_strength_proxy"].apply(
            lambda x: "weak" if x > med else "strong"
        )
        print(f"  tie_type column absent — derived locally "
              f"(median proxy = {med}, Module 1 rule).")

    # Temporal features: reuse if the cleaned file already has them, else compute.
    if "year_month" not in c.columns:
        c["year_month"] = c["comment_created_utc"].dt.to_period("M").astype(str)
    if "year" not in c.columns:
        c["year"] = c["comment_created_utc"].dt.year
    if "month" not in c.columns:
        c["month"] = c["comment_created_utc"].dt.month

    # Thematic-only: comment length and the substantive-comment filter.
    c["word_count"] = (c["comment_body"].astype(str)
                       .str.split().str.len().fillna(0).astype(int))
    c = c[c["word_count"] >= MIN_WORDS].copy()

    if sample_n:
        c = c.sample(min(sample_n, len(c)), random_state=RANDOM_SEED)

    print(f"  Loaded {len(c):,} substantive comments (≥{MIN_WORDS} words)")
    return c.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> tuple[list[list[str]], list[str]]:
    """Tokenize all comments. Returns token lists and joined strings."""
    print("Preprocessing...")
    token_lists, joined = [], []
    for text in df["comment_body"]:
        toks = tokenize(str(text))
        token_lists.append(toks)
        joined.append(" ".join(toks))
    return token_lists, joined


# ════════════════════════════════════════════════════════════════════════════════
# METHOD 1: TF-IDF + NMF
# ════════════════════════════════════════════════════════════════════════════════
def run_nmf(joined_texts: list[str], n_topics: int = N_TOPICS
            ) -> tuple[np.ndarray, np.ndarray, list[str], TfidfVectorizer]:
    """
    Non-negative Matrix Factorization on TF-IDF matrix.
    Fast, interpretable, handles short texts well.
    Returns: doc-topic matrix W, topic-term matrix H, feature names, vectorizer.
    """
    print(f"Running TF-IDF + NMF ({n_topics} topics)...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_VOCAB,
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
        token_pattern=r"[a-z][a-z]+",
    )
    X = vectorizer.fit_transform(joined_texts)
    nmf = NMF(
        n_components=n_topics,
        random_state=RANDOM_SEED,
        max_iter=500,
        init="nndsvda",
    )
    W = nmf.fit_transform(X)
    H = nmf.components_
    feat_names = vectorizer.get_feature_names_out().tolist()
    print(f"  TF-IDF matrix: {X.shape}  →  NMF W: {W.shape}")
    return W, H, feat_names, vectorizer


# ════════════════════════════════════════════════════════════════════════════════
# METHOD 2: LDA (Gensim)
# ════════════════════════════════════════════════════════════════════════════════
def run_lda(token_lists: list[list[str]], n_topics: int = N_TOPICS
            ) -> tuple[LdaModel, corpora.Dictionary, list, float]:
    """
    Latent Dirichlet Allocation via Gensim.
    Probabilistic generative model; supports coherence scoring.
    Returns: model, dictionary, corpus, coherence score.
    """
    print(f"Running LDA ({n_topics} topics, {LDA_PASSES} passes)...")
    # Filter empty token lists
    token_lists = [t for t in token_lists if len(t) > 0]
    dictionary = corpora.Dictionary(token_lists)
    dictionary.filter_extremes(no_below=5, no_above=0.85, keep_n=MAX_VOCAB)
    corpus = [dictionary.doc2bow(t) for t in token_lists]

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=RANDOM_SEED,
        passes=LDA_PASSES,
        iterations=LDA_ITER,
        alpha="auto",
        eta="auto",
        per_word_topics=False,
    )

    # Coherence (c_v metric — higher is better, ~0.4–0.6 is good)
    try:
        coh = CoherenceModel(
            model=model, texts=token_lists,
            dictionary=dictionary, coherence="c_v"
        ).get_coherence()
    except Exception:
        coh = float("nan")

    print(f"  LDA coherence (c_v): {coh:.4f}")
    return model, dictionary, corpus, coh


# ════════════════════════════════════════════════════════════════════════════════
# METHOD 3: RAKE keyword extraction
# ════════════════════════════════════════════════════════════════════════════════
def rake_keywords(texts: list[str], top_n: int = 20) -> list[tuple[str, float]]:
    """
    Rapid Automatic Keyword Extraction (RAKE).
    Extracts multi-word phrases without a pre-trained model.
    """
    stop_set = ALL_STOPS

    def score_phrase(phrase: str) -> float:
        words = phrase.split()
        if len(words) == 0:
            return 0.0
        freq = Counter(words)
        degree = sum(len(phrase.split()) for phrase in [phrase])
        return sum((degree / freq[w]) for w in words)

    phrase_freq: Counter = Counter()
    phrase_score: dict = {}

    for text in texts:
        text_clean = clean_text(str(text))
        # Split on stop words to get candidate phrases
        stop_pattern = r"\b(?:" + "|".join(re.escape(s) for s in stop_set) + r")\b"
        phrases = re.split(stop_pattern, text_clean)
        for ph in phrases:
            ph = ph.strip()
            words = [w for w in ph.split() if w.isalpha() and len(w) > 2]
            if 1 <= len(words) <= 4:
                key = " ".join(words)
                phrase_freq[key] += 1
                phrase_score[key] = score_phrase(key) * phrase_freq[key]

    return sorted(phrase_score.items(), key=lambda x: -x[1])[:top_n]


# ════════════════════════════════════════════════════════════════════════════════
# TOPIC LABELING AND DOMAIN MAPPING
# ════════════════════════════════════════════════════════════════════════════════
def get_top_words(H: np.ndarray, feat_names: list[str], n: int = 12) -> list[list[str]]:
    """Extract top-n words for each NMF topic."""
    return [
        [feat_names[j] for j in row.argsort()[-n:][::-1]]
        for row in H
    ]


def align_to_codebook(top_words: list[str], n_consider: int = 12) -> dict:
    """
    Post-hoc alignment of an ALREADY-discovered topic to the a priori codebook.

    This does not influence topic discovery (NMF/LDA are unsupervised). It measures
    overlap between a topic's keywords and each lexically-seeded subtheme, then
    reports the alignment transparently instead of forcing a nearest bucket:

      status = "aligned"             clear, multi-term match to one subtheme
             = "ambiguous"           weak (single term) or cross-domain near-tie
             = "inductive_candidate" no seed overlap -> feeds human inductive coding
    """
    words = list(dict.fromkeys([w for w in top_words[:n_consider]]))  # ordered, unique
    rank = {w: i for i, w in enumerate(words)}

    hits = []
    for code in LEXICAL_SUBTHEMES:
        matched = [w for w in words if w in SEEDSETS[code]]
        if matched:
            overlap = len(matched)
            weighted = round(sum(1.0 / (rank[w] + 1) for w in matched), 3)
            hits.append((code, overlap, weighted, matched))

    base = {
        "subtheme_code": "?", "subtheme_label": DOMAIN_LABELS["?"],
        "domain_code": "?",   "domain_label": DOMAIN_LABELS["?"],
        "status": "inductive_candidate",
        "overlap": 0, "alignment_strength": 0.0, "matched_terms": "",
        "runner_up_code": "", "runner_up_overlap": 0, "runner_up_domain": "",
    }
    if not hits:
        return base

    hits.sort(key=lambda h: (h[1], h[2]), reverse=True)
    best = hits[0]
    runner = hits[1] if len(hits) > 1 else None

    b_code, b_overlap, b_weighted, b_matched = best
    b_label, b_domain, _, _ = CODEBOOK[b_code]

    r_code = r_overlap = r_domain = None
    clear = True
    if runner is not None:
        r_code, r_overlap, r_weighted, _ = runner
        _, r_domain, _, _ = CODEBOOK[r_code]
        if r_domain != b_domain and (b_overlap - r_overlap) < AMBIG_MARGIN:
            clear = False  # a different domain is essentially tied for best

    if b_overlap >= MIN_ALIGN_OVERLAP and clear:
        status = "aligned"
    elif b_overlap >= 1:
        status = "ambiguous"
    else:
        status = "inductive_candidate"

    inductive = status == "inductive_candidate"
    return {
        "subtheme_code":  "?" if inductive else b_code,
        "subtheme_label": DOMAIN_LABELS["?"] if inductive else b_label,
        "domain_code":    "?" if inductive else b_domain,
        "domain_label":   DOMAIN_LABELS["?"] if inductive else DOMAIN_LABELS[b_domain],
        "status":         status,
        "overlap":        b_overlap,
        "alignment_strength": round(b_overlap / max(1, len(words)), 2),
        "matched_terms":  ", ".join(b_matched),
        "runner_up_code":    r_code or "",
        "runner_up_overlap": r_overlap or 0,
        "runner_up_domain":  r_domain or "",
    }


def label_topics(H: np.ndarray, feat_names: list[str],
                 lda_model: LdaModel | None = None,
                 n_top_words: int = 12
                 ) -> pd.DataFrame:
    """
    Build a topic summary with keywords and a TRANSPARENT codebook alignment.
    The codebook is applied post-hoc (align_to_codebook); topics are never forced
    into a domain. Records alignment status, overlap strength, matched terms and a
    cross-domain runner-up so weak/ambiguous mappings are visible, not hidden.
    """
    top_words_nmf = get_top_words(H, feat_names, n=n_top_words)

    rows = []
    for i, words in enumerate(top_words_nmf):
        a = align_to_codebook(words)

        # NOTE: lda_words is the LDA topic at the SAME index, which is an arbitrary
        # ordering unrelated to NMF topic i. Kept only as a loose keyword reference,
        # not a cross-method validation (that would require matching topics by
        # word-distribution similarity first).
        lda_words = []
        if lda_model and i < lda_model.num_topics:
            lda_words = [w for w, _ in lda_model.show_topic(i, topn=10)]

        rows.append({
            "topic_id":          i,
            "topic_label":       f"T{i:02d}",
            "domain_code":       a["domain_code"],
            "domain_label":      a["domain_label"],
            "subtheme_code":     a["subtheme_code"],
            "subtheme_label":    a["subtheme_label"],
            "alignment_status":  a["status"],
            "overlap":           a["overlap"],
            "alignment_strength": a["alignment_strength"],
            "matched_terms":     a["matched_terms"],
            "runner_up_code":    a["runner_up_code"],
            "runner_up_overlap": a["runner_up_overlap"],
            "runner_up_domain":  a["runner_up_domain"],
            "nmf_keywords":      " | ".join(words),
            "lda_keywords":      " | ".join(lda_words) if lda_words else "",
            "top_5_nmf":         ", ".join(words[:5]),
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════════
# ASSIGNMENT AND CROSS-TABLES
# ════════════════════════════════════════════════════════════════════════════════
def assign_topics(df: pd.DataFrame, W: np.ndarray,
                  topic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each comment its dominant topic and domain.
    Attaches to the original dataframe.
    """
    df = df.copy()
    df["dominant_topic_id"]    = W.argmax(axis=1)
    df["topic_confidence"]     = W.max(axis=1)
    df["topic_label"]          = df["dominant_topic_id"].map(
        dict(zip(topic_df["topic_id"], topic_df["topic_label"])))
    df["domain_code"]          = df["dominant_topic_id"].map(
        dict(zip(topic_df["topic_id"], topic_df["domain_code"])))
    df["domain_label"]         = df["dominant_topic_id"].map(
        dict(zip(topic_df["topic_id"], topic_df["domain_label"])))
    df["top_5_nmf"]            = df["dominant_topic_id"].map(
        dict(zip(topic_df["topic_id"], topic_df["top_5_nmf"])))
    # Machine-suggested codebook alignment (a coding AID, not a code). Prefixed
    # auto_ so it can be dropped before a coder codes fresh (per RQ2 protocol).
    df["auto_subtheme_code"]    = df["dominant_topic_id"].map(
        dict(zip(topic_df["topic_id"], topic_df["subtheme_code"])))
    df["auto_alignment_status"] = df["dominant_topic_id"].map(
        dict(zip(topic_df["topic_id"], topic_df["alignment_status"])))
    return df


def cross_table(df: pd.DataFrame, col: str, normalize: bool = True) -> pd.DataFrame:
    """Cross-tabulate topic vs a categorical column."""
    ct = pd.crosstab(df["topic_label"], df[col])
    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0).round(3) * 100
    return ct


# ════════════════════════════════════════════════════════════════════════════════
# RAKE PER-DOMAIN KEYWORDS
# ════════════════════════════════════════════════════════════════════════════════
def rake_per_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Run RAKE on each domain's assigned comments to extract top phrases."""
    rows = []
    for domain in sorted(df["domain_code"].dropna().unique()):
        texts = df[df["domain_code"] == domain]["comment_body"].tolist()
        if len(texts) < 10:
            continue
        phrases = rake_keywords(texts, top_n=15)
        for phrase, score in phrases:
            rows.append({
                "domain_code":  domain,
                "phrase":       phrase,
                "rake_score":   round(score, 2),
                "n_comments_in_domain": len(texts),
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════════
def fig_topic_overview(topic_df: pd.DataFrame, df: pd.DataFrame):
    """Bar chart: comment count per topic, colored by domain."""
    counts = df["topic_label"].value_counts().sort_index()
    domain_colors = {"A": "#dc3545", "B": "#0d6efd", "C": "#fd7e14",
                     "D": "#198754", "?": "#adb5bd"}
    colors = [
        domain_colors.get(topic_df.set_index("topic_label").loc[t, "domain_code"], "#adb5bd")
        if t in topic_df["topic_label"].values else "#adb5bd"
        for t in counts.index
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor(SF)
    bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Comment count", fontsize=9, color=MU)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f"{int(bar.get_height()):,}",
                ha="center", va="bottom", fontsize=7.5, color=TX)

    handles = [mpatches.Patch(color=v, label=f"Domain {k}")
               for k, v in domain_colors.items() if k != "?"]
    ax.legend(handles=handles, fontsize=8, facecolor="white",
              edgecolor=GR, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "theme_topic_overview.png"), **SAVE)
    plt.close()
    print("  theme_topic_overview.png")


def fig_topic_by_subreddit(df: pd.DataFrame):
    """Heatmap of topic distribution per subreddit (% of comments)."""
    ct = pd.crosstab(df["subreddit"], df["topic_label"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_facecolor(BG)
    im = ax.imshow(ct_pct.values, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(len(ct_pct.columns)))
    ax.set_xticklabels(ct_pct.columns, fontsize=8, rotation=45, ha="right", color=TX)
    ax.set_yticks(range(len(ct_pct.index)))
    ax.set_yticklabels([f"r/{s}" for s in ct_pct.index], fontsize=9, color=TX)
    for i in range(len(ct_pct.index)):
        for j in range(len(ct_pct.columns)):
            val = ct_pct.values[i, j]
            if val > 3:
                ax.text(j, i, f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=6.5,
                        color="white" if val > 12 else "#212529")
    plt.colorbar(im, ax=ax, label="% of subreddit comments", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "theme_by_subreddit.png"), **SAVE)
    plt.close()
    print("  theme_by_subreddit.png")


def fig_topic_by_tie_type(df: pd.DataFrame):
    """Grouped bar: topic distribution by tie type — the Granovetter lens."""
    ct = pd.crosstab(df["topic_label"], df["tie_type"], normalize="index") * 100
    if "weak" not in ct.columns:
        ct["weak"] = 0
    if "strong" not in ct.columns:
        ct["strong"] = 0

    x = np.arange(len(ct))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor(SF)
    ax.bar(x - w / 2, ct["weak"],   w, color=TIE_C["weak"],
           alpha=0.88, label="Weak-tie", edgecolor="white", lw=0.5)
    ax.bar(x + w / 2, ct["strong"], w, color=TIE_C["strong"],
           alpha=0.88, label="Strong-tie", edgecolor="white", lw=0.5)
    ax.axhline(50, color=MU, lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ct.index, fontsize=8, rotation=45, ha="right", color=TX)
    ax.set_ylabel("% of topic's comments", fontsize=9, color=MU)
    ax.legend(fontsize=9, facecolor="white", edgecolor=GR)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "theme_by_tie_type.png"), **SAVE)
    plt.close()
    print("  theme_by_tie_type.png")


def fig_topic_keywords(topic_df: pd.DataFrame):
    """Table-style figure showing keywords per topic."""
    n = len(topic_df)
    fig, ax = plt.subplots(figsize=(14, n * 0.52 + 1))
    ax.set_facecolor(BG)
    ax.axis("off")

    domain_colors = {"A": "#dc3545", "B": "#0d6efd", "C": "#fd7e14",
                     "D": "#198754", "?": "#adb5bd"}
    col_widths = [0.06, 0.08, 0.86]
    col_xs = [0.01, 0.08, 0.17]
    headers = ["Topic", "Domain", "Top keywords"]
    for j, (header, x) in enumerate(zip(headers, col_xs)):
        ax.text(x, 1.0, header, transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="#212529", va="top")

    for i, row in topic_df.iterrows():
        y = 1.0 - (i + 1.5) / (n + 1.5)
        bg_color = "#f8f9fa" if i % 2 == 0 else "#ffffff"
        ax.axhspan(y - 0.5 / (n + 2), y + 0.5 / (n + 2),
                   xmin=0, xmax=1, color=bg_color, zorder=0)

        ax.text(col_xs[0], y, row["topic_label"], transform=ax.transAxes,
                fontsize=8.5, va="center", color="#212529")

        dc = row["domain_code"]
        dc_color = domain_colors.get(dc, "#adb5bd")
        ax.text(col_xs[1], y, dc, transform=ax.transAxes,
                fontsize=8.5, va="center", color=dc_color, fontweight="bold")

        ax.text(col_xs[2], y, row["nmf_keywords"][:80],
                transform=ax.transAxes,
                fontsize=7.5, va="center", color="#495057")

    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(FIG_DIR, "theme_keywords_table.png"), **SAVE)
    plt.close()
    print("  theme_keywords_table.png")


def fig_temporal_topics(df: pd.DataFrame, top_topics: int = 5):
    """Line chart: volume of top topics over time."""
    top = df["topic_label"].value_counts().head(top_topics).index.tolist()
    monthly = (df[df["topic_label"].isin(top)]
               .groupby(["year_month", "topic_label"])
               .size().unstack(fill_value=0))

    months_str = monthly.index.tolist()
    x = np.arange(len(months_str))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top)))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_facecolor(SF)
    for topic, color in zip(top, colors):
        if topic in monthly.columns:
            ax.plot(x, monthly[topic], lw=1.8, label=topic, color=color, zorder=3)
            ax.fill_between(x, monthly[topic], alpha=0.06, color=color)

    tick_step = max(1, len(months_str) // 12)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(months_str[::tick_step], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Comments per month", fontsize=9, color=MU)
    ax.legend(fontsize=8, facecolor="white", edgecolor=GR, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "theme_temporal.png"), **SAVE)
    plt.close()
    print("  theme_temporal.png")


def fig_domain_donut(df: pd.DataFrame):
    """Donut chart: proportion of comments per domain."""
    domain_labels = {
        "A": "A — Faculty Evaluation",
        "B": "B — Institutional",
        "C": "C — Evaluative Rhetoric",
        "D": "D — Community Norms",
        "?": "Unclassified",
    }
    domain_colors_map = {
        "A": "#dc3545", "B": "#0d6efd",
        "C": "#fd7e14", "D": "#198754", "?": "#adb5bd",
    }
    counts = df["domain_code"].value_counts()
    labels = [domain_labels.get(k, k) for k in counts.index]
    colors = [domain_colors_map.get(k, "#adb5bd") for k in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, colors=colors,
        startangle=90, autopct="%1.1f%%", pctdistance=0.72,
        wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2),
    )
    for t in texts: t.set_color(TX); t.set_fontsize(9)
    for t in autotexts: t.set_color("white"); t.set_fontsize(9); t.set_fontweight("bold")
    ax.text(0, 0, f"n={len(df):,}", ha="center", va="center",
            fontsize=11, fontweight="bold", color=TX)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "theme_domain_donut.png"), **SAVE)
    plt.close()
    print("  theme_domain_donut.png")


# ════════════════════════════════════════════════════════════════════════════════
# NVIVO EXPORT
# ════════════════════════════════════════════════════════════════════════════════
def build_nvivo_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build NVivo-ready export. Retains all original columns plus
    computational theme assignments. Blank coding columns are preserved
    from the qualitative datasets for manual review.
    """
    CODING_COLS = [
        "CODER_INITIALS", "CODING_DATE",
        "DOMAIN_A_CODE", "DOMAIN_A_SUBTHEME",
        "DOMAIN_B_CODE", "DOMAIN_B_SUBTHEME",
        "DOMAIN_C_CODE", "DOMAIN_C_SUBTHEME",
        "DOMAIN_D_CODE", "DOMAIN_D_SUBTHEME",
        "SENTIMENT_HUMAN", "SENTIMENT_MATCH_MODEL",
        "IS_SARCASM", "IS_TAGLISH", "CODER_NOTES",
    ]
    KEEP = [
        "k",
        "comment_id", "subreddit", "post_title", "comment_author",
        "comment_body", "comment_score", "comment_depth",
        "comment_created_utc", "tie_type", "tie_strength_proxy",
        "dominant_topic_id", "topic_label", "topic_confidence",
        "domain_code", "domain_label", "top_5_nmf",
        "auto_subtheme_code", "auto_alignment_status",
    ]
    out = df[[c for c in KEEP if c in df.columns]].copy()
    for col in CODING_COLS:
        out[col] = ""
    return out


# ════════════════════════════════════════════════════════════════════════════════
# CODEBOOK ALIGNMENT OUTPUTS (transparent, post-hoc)
# ════════════════════════════════════════════════════════════════════════════════
def build_codebook_alignment_outputs(topic_df: pd.DataFrame,
                                     df_full: pd.DataFrame
                                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produce three audit artefacts that make the codebook alignment a measured
    result rather than a forced assignment:

      1. topic_codebook_alignment.csv — every topic, its alignment status, overlap
         strength, matched seed terms, and cross-domain runner-up.
      2. inductive_candidates.csv — topics that did NOT align cleanly, with exemplar
         comments, to feed the human inductive-coding pass (Fereday & Muir-Cochrane
         step 4: identify data-driven codes the template missed).
      3. codebook_coverage.csv — for each codebook subtheme, how many emergent
         topics/comments aligned to it; i.e. how well the corpus recovers the a
         priori template, reported as a finding.
    """
    # 1) per-topic alignment table
    align_cols = ["topic_id", "topic_label", "alignment_status", "domain_code",
                  "domain_label", "subtheme_code", "subtheme_label", "overlap",
                  "alignment_strength", "matched_terms", "runner_up_code",
                  "runner_up_overlap", "runner_up_domain", "top_5_nmf",
                  "nmf_keywords"]
    alignment = topic_df[[c for c in align_cols if c in topic_df.columns]].copy()

    # 2) inductive candidates + exemplars
    cand_rows = []
    nonaligned = topic_df[topic_df["alignment_status"] != "aligned"]
    has_body = "comment_body" in df_full.columns
    for _, t in nonaligned.iterrows():
        tid = t["topic_id"]
        sub = df_full[df_full["dominant_topic_id"] == tid]
        exemplars = []
        if has_body and len(sub):
            exemplars = (sub.nlargest(3, "topic_confidence")["comment_body"]
                         .astype(str).str.replace(r"\s+", " ", regex=True)
                         .str.slice(0, 160).tolist())
        cand_rows.append({
            "topic_id": tid,
            "topic_label": t["topic_label"],
            "alignment_status": t["alignment_status"],
            "nearest_subtheme": (f'{t["subtheme_code"]} {t["subtheme_label"]}'
                                 if t["subtheme_code"] != "?" else "(none)"),
            "overlap": t["overlap"],
            "matched_terms": t["matched_terms"],
            "n_comments": int((df_full["dominant_topic_id"] == tid).sum()),
            "keywords": t["nmf_keywords"],
            "exemplar_1": exemplars[0] if len(exemplars) > 0 else "",
            "exemplar_2": exemplars[1] if len(exemplars) > 1 else "",
            "exemplar_3": exemplars[2] if len(exemplars) > 2 else "",
        })
    candidates = pd.DataFrame(cand_rows)

    # 3) codebook coverage
    have_auto = {"auto_subtheme_code", "auto_alignment_status"}.issubset(df_full.columns)
    aligned_topics = topic_df[topic_df["alignment_status"] == "aligned"]
    cov_rows = []
    for code, (label, dom, lex, seeds) in CODEBOOK.items():
        n_topics = int((aligned_topics["subtheme_code"] == code).sum())
        if have_auto:
            mask = ((df_full["auto_subtheme_code"] == code) &
                    (df_full["auto_alignment_status"] == "aligned"))
            n_comments = int(mask.sum())
        else:
            n_comments = 0
        cov_rows.append({
            "subtheme_code": code,
            "subtheme_label": label,
            "domain_code": dom,
            "seeded": "yes" if lex else "human-coded only",
            "topics_aligned": n_topics,
            "comments_aligned": n_comments,
            "recovered_computationally": ("yes" if n_topics > 0
                                          else ("no" if lex else "n/a — human-coded")),
        })
    coverage = pd.DataFrame(cov_rows)
    return alignment, candidates, coverage


# ════════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════════
def run_for_k(df: pd.DataFrame,
              token_lists: list[list[str]],
              joined_texts: list[str],
              k: int,
              output_root: str) -> dict:
    """
    Run the full NMF + LDA + labeling + export + figures pipeline for a single
    topic count k. All artefacts are written to {output_root}/k{k}/.

    The loaded corpus and tokenization are computed once by the caller and
    reused across every k, so only the modeling steps re-run here.

    Returns a one-row summary dict for the cross-k comparison table.
    """
    global OUT_DIR, FIG_DIR, MODEL_DIR
    OUT_DIR   = os.path.join(output_root, f"k{k}")
    FIG_DIR   = os.path.join(OUT_DIR, "figures")
    MODEL_DIR = os.path.join(OUT_DIR, "lda_model")
    for d in (OUT_DIR, FIG_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)

    print(f"\n{'#'*64}")
    print(f"# k = {k} topics   →   {OUT_DIR}")
    print(f"{'#'*64}")

    # ── NMF ───────────────────────────────────────────────────────────────────
    W, H, feat_names, vectorizer = run_nmf(joined_texts, n_topics=k)

    # ── LDA ───────────────────────────────────────────────────────────────────
    lda_model, lda_dict, lda_corpus, coherence = run_lda(token_lists, n_topics=k)
    lda_model.save(os.path.join(MODEL_DIR, "lda.model"))
    lda_dict.save(os.path.join(MODEL_DIR, "lda_dictionary"))
    print(f"  LDA model saved to {MODEL_DIR}/")

    # ── Topic labeling ────────────────────────────────────────────────────────
    topic_df = label_topics(H, feat_names, lda_model=lda_model)
    topic_df["lda_coherence"] = coherence
    topic_df.insert(0, "k", k)
    print("\n── TOPIC SUMMARY ──")
    for _, row in topic_df.iterrows():
        print(f"  {row['topic_label']} [{row['domain_code']}] {row['domain_label']}")
        print(f"    Keywords: {row['top_5_nmf']}")

    # ── Assign topics to comments ─────────────────────────────────────────────
    df_full = assign_topics(df, W, topic_df)
    df_full.insert(0, "k", k)

    # ── RAKE per domain ───────────────────────────────────────────────────────
    print("\nRunning RAKE keyword extraction per domain...")
    rake_df = rake_per_domain(df_full)

    # ── Cross-tables ──────────────────────────────────────────────────────────
    topic_by_sub = cross_table(df_full, "subreddit")
    topic_by_tie = cross_table(df_full, "tie_type")
    topic_by_month = (
        df_full.groupby(["year_month", "topic_label"])
        .size().unstack(fill_value=0)
        .reset_index()
    )

    # ── Topic counts → summary ────────────────────────────────────────────────
    counts = df_full["topic_label"].value_counts()
    topic_df["n_comments"] = (
        counts.reindex(topic_df["topic_label"]).fillna(0).astype(int).values
    )
    topic_df["pct_of_corpus"] = (
        topic_df["n_comments"] / len(df_full) * 100
    ).round(1)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving outputs...")
    topic_df.to_csv(os.path.join(OUT_DIR, "theme_summary.csv"), index=False)
    df_full.to_csv(os.path.join(OUT_DIR, "comments_with_topics.csv"), index=False)
    topic_by_sub.to_csv(os.path.join(OUT_DIR, "topic_by_subreddit.csv"))
    topic_by_tie.to_csv(os.path.join(OUT_DIR, "topic_by_tie_type.csv"))
    topic_by_month.to_csv(os.path.join(OUT_DIR, "topic_by_month.csv"), index=False)
    rake_df.to_csv(os.path.join(OUT_DIR, "rake_keywords_by_domain.csv"), index=False)

    nvivo_export = build_nvivo_export(df_full)
    nvivo_export.to_csv(
        os.path.join(OUT_DIR, "nvivo_thematic_export.csv"),
        index=False, encoding="utf-8-sig",
    )

    # Codebook alignment artefacts (transparent, post-hoc)
    alignment, candidates, coverage = build_codebook_alignment_outputs(topic_df, df_full)
    alignment.to_csv(os.path.join(OUT_DIR, "topic_codebook_alignment.csv"), index=False)
    candidates.to_csv(os.path.join(OUT_DIR, "inductive_candidates.csv"),
                      index=False, encoding="utf-8-sig")
    coverage.to_csv(os.path.join(OUT_DIR, "codebook_coverage.csv"), index=False)
    print(f"  All CSVs saved to {OUT_DIR}/")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_topic_overview(topic_df, df_full)
    fig_topic_by_subreddit(df_full)
    fig_topic_by_tie_type(df_full)
    fig_topic_keywords(topic_df)
    fig_temporal_topics(df_full)
    fig_domain_donut(df_full)

    # ── Per-k console summary ─────────────────────────────────────────────────
    n_aligned = int((topic_df["alignment_status"] == "aligned").sum())
    n_ambig   = int((topic_df["alignment_status"] == "ambiguous").sum())
    n_induct  = int((topic_df["alignment_status"] == "inductive_candidate").sum())
    print(f"\nCodebook alignment (k={k}): {n_aligned} aligned · "
          f"{n_ambig} ambiguous · {n_induct} inductive candidate "
          f"of {len(topic_df)} topics")
    print("  (ambiguous + inductive topics are listed in inductive_candidates.csv "
          "for the human inductive pass)")

    print(f"\nk={k} domain distribution (auto-alignment):")
    domain_counts = df_full["domain_code"].value_counts()
    for domain, count in domain_counts.items():
        pct = count / len(df_full) * 100
        dlabel = DOMAIN_LABELS.get(domain, domain)
        print(f"  {domain} {dlabel}: {count:,} comments ({pct:.1f}%)")

    return {
        "k": k,
        "lda_coherence": coherence,
        "n_comments": len(df_full),
        "topics_aligned": n_aligned,
        "topics_ambiguous": n_ambig,
        "topics_inductive": n_induct,
        "pct_comments_inductive": round((df_full["domain_code"] == "?").mean() * 100, 1),
        "output_dir": OUT_DIR,
    }


def write_k_comparison(summaries: list[dict], output_root: str) -> str:
    """Write a cross-k comparison CSV + coherence-vs-k figure at the run root."""
    comp = pd.DataFrame(summaries)
    comp_path = os.path.join(output_root, "k_comparison.csv")
    comp.to_csv(comp_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor(SF)
    ax.plot(comp["k"], comp["lda_coherence"], "-o", color="#0d6efd",
            linewidth=2, markersize=8, markeredgecolor="white", zorder=3)
    for _, r in comp.iterrows():
        if pd.notna(r["lda_coherence"]):
            ax.annotate(f"{r['lda_coherence']:.3f}",
                        (r["k"], r["lda_coherence"]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=TX)
    ax.set_xlabel("Number of topics (k)", fontsize=9, color=MU)
    ax.set_ylabel("LDA coherence (c_v)", fontsize=9, color=MU)
    ax.set_xticks(comp["k"])
    ax.set_title("Topic coherence by k (higher = better)", fontsize=11,
                 color=TX, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "k_comparison_coherence.png"), **SAVE)
    plt.close()
    print(f"  k_comparison.csv + k_comparison_coherence.png")
    return comp_path


def regenerate_figures(output_root: str):
    """Regenerate figures for every k* subdir under an existing outputs/ dir."""
    global OUT_DIR, FIG_DIR, MODEL_DIR
    k_dirs = sorted(
        d for d in os.listdir(output_root)
        if re.fullmatch(r"k\d+", d) and os.path.isdir(os.path.join(output_root, d))
    )
    if not k_dirs:
        sys.exit(f"No k* subdirectories found in {output_root}")
    for kd in k_dirs:
        OUT_DIR   = os.path.join(output_root, kd)
        FIG_DIR   = os.path.join(OUT_DIR, "figures")
        MODEL_DIR = os.path.join(OUT_DIR, "lda_model")
        os.makedirs(FIG_DIR, exist_ok=True)
        results_path = os.path.join(OUT_DIR, "comments_with_topics.csv")
        summary_path = os.path.join(OUT_DIR, "theme_summary.csv")
        if not (os.path.exists(results_path) and os.path.exists(summary_path)):
            print(f"  skipping {kd}: missing results CSVs")
            continue
        print(f"Regenerating figures for {kd}...")
        df_full = pd.read_csv(results_path, parse_dates=["comment_created_utc"])
        topic_df = pd.read_csv(summary_path)
        fig_topic_overview(topic_df, df_full)
        fig_topic_by_subreddit(df_full)
        fig_topic_by_tie_type(df_full)
        fig_topic_keywords(topic_df)
        fig_temporal_topics(df_full)
        fig_domain_donut(df_full)
    print("Done.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="HEI thematic analysis pipeline (runs multiple k values)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to N comments (for quick testing)")
    parser.add_argument("--k-values", type=str, default=",".join(map(str, K_VALUES)),
                        help="Comma-separated topic counts to run (default: 10,15,20)")
    parser.add_argument("--only-figures", action="store_true",
                        help="Regenerate figures from an existing run "
                             "(requires --run-dir)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Existing .../outputs dir to use with --only-figures")
    # parse_known_args ignores extras the Jupyter/Colab kernel injects into
    # sys.argv (e.g. "-f /root/.local/.../kernel.json") when main() is called
    # from inside a notebook cell rather than the command line.
    args, _unknown = parser.parse_known_args(argv)

    # ── only-figures mode ─────────────────────────────────────────────────────
    if args.only_figures:
        run_dir = args.run_dir
        if not run_dir or not os.path.isdir(run_dir):
            sys.exit("--only-figures requires --run-dir pointing to an existing "
                     "outputs/ directory (the folder containing k10/, k15/, ...)")
        regenerate_figures(run_dir)
        return

    k_values = [int(x) for x in str(args.k_values).split(",") if str(x).strip()]
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"Source dir:  {DATA_DIR}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"k values:    {k_values}")

    # ── Load + preprocess ONCE (shared across every k) ────────────────────────
    df = load_data(sample_n=args.sample)
    token_lists, joined_texts = preprocess(df)

    # ── Run each k into its own k{K}/ subfolder ───────────────────────────────
    summaries = []
    for k in k_values:
        summaries.append(run_for_k(df, token_lists, joined_texts, k, OUTPUT_ROOT))

    # ── Cross-k comparison ────────────────────────────────────────────────────
    print("\nWriting cross-k comparison...")
    comp_path = write_k_comparison(summaries, OUTPUT_ROOT)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("ALL RUNS COMPLETE")
    print(f"{'='*64}")
    print(f"Output root: {OUTPUT_ROOT}\n")
    print(f"Coherence + codebook alignment by k:")
    for s in summaries:
        coh = s["lda_coherence"]
        coh_str = f"{coh:.4f}" if coh == coh else "  n/a "   # NaN-safe
        print(f"  k={s['k']:>2}:  c_v={coh_str}   "
              f"aligned={s['topics_aligned']:>2} "
              f"ambiguous={s['topics_ambiguous']:>2} "
              f"inductive={s['topics_inductive']:>2}   →  {s['output_dir']}")
    print(f"\nComparison table: {comp_path}")


if __name__ == "__main__":
    main()
