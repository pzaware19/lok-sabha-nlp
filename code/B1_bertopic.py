"""
B1_bertopic.py — BERTopic topic model on Lok Sabha starred questions
Author: Piyush Zaware
Updated: 2026-06-12

Runs BERTopic on individual starred questions (not party-session aggregates).
Uses paraphrase-multilingual-MiniLM-L12-v2 — fast, multilingual, handles
Indian English and transliterated terms in parliamentary language.

Inputs:
    tmp/train-*.parquet          — raw question files (from A1)
    tmp/party_lookup.csv         — MP → party mapping (from A2)

Outputs:
    output/tables/bertopic_topics.csv         — topic id, top words, count
    output/tables/bertopic_doc_assignments.csv — question id, topic, party, lok_no
    output/tables/bertopic_party_matrix.csv   — party × topic prevalence matrix
    output/models/bertopic_model/             — serialized model
"""

import os, glob, warnings
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT    = "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
TMPDIR  = os.path.join(ROOT, "tmp")
OUTDIR  = os.path.join(ROOT, "output")
TABDIR  = os.path.join(OUTDIR, "tables")
MODDIR  = os.path.join(OUTDIR, "models")

os.makedirs(TABDIR, exist_ok=True)
os.makedirs(MODDIR, exist_ok=True)

# ── 1. Load starred questions ──────────────────────────────────────────────

print("Loading parquet files...")
parquet_files = sorted(glob.glob(os.path.join(TMPDIR, "train-*.parquet")))
dfs = [pd.read_parquet(f, columns=["id", "lok_no", "session_no", "type",
                                    "ministry", "members", "question_text"])
       for f in parquet_files]
raw = pd.concat(dfs, ignore_index=True)
print(f"  Total rows: {len(raw):,}")

starred = raw[raw["type"] == "STARRED"].copy()
print(f"  Starred questions: {len(starred):,}")

# ── 2. Load party lookup and merge ─────────────────────────────────────────
# party_lookup maps cleaned MP name → party_family
# Build it from the party_session metadata + original MP matching

lookup_path = os.path.join(TMPDIR, "party_lookup.csv")
if os.path.exists(lookup_path):
    lookup = pd.read_csv(lookup_path)
    print(f"  Party lookup rows: {len(lookup):,}")
else:
    # Fall back: use doc_meta_mp.csv which has primary_mp → party_family
    lookup = pd.read_csv(os.path.join(TMPDIR, "doc_meta_mp.csv"))
    lookup = lookup.rename(columns={"primary_mp": "mp_clean", "doc_mp": "mp_raw"})
    print(f"  Using doc_meta_mp.csv: {len(lookup):,} rows")

# Normalize MP names for matching
def norm(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).upper().strip()

# members is an Arrow list column — each row is a list of MP name strings
# Extract first element safely
def first_member(x):
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return norm(x[0])
    if isinstance(x, str) and len(x) > 0:
        return norm(x.split(";")[0].split("\n")[0].split(",")[0])
    return ""

starred["primary_member"] = starred["members"].apply(first_member)

# Try to match against lookup
if "mp_raw" in lookup.columns:
    lookup["mp_key"] = lookup["mp_raw"].apply(norm)
elif "mp_clean" in lookup.columns:
    lookup["mp_key"] = lookup["mp_clean"].apply(norm)
else:
    lookup["mp_key"] = lookup.iloc[:, 0].apply(norm)

mp_to_party = dict(zip(lookup["mp_key"], lookup["party_family"]))

starred["party_family"] = starred["primary_member"].map(mp_to_party)
matched = starred.dropna(subset=["party_family"])
print(f"  Matched to party: {len(matched):,} ({100*len(matched)/len(starred):.1f}%)")

# If match rate is very low, fall back to all starred questions for topic model
# and mark unmatched party as "Unknown"
if len(matched) < 500:
    print("  Low match rate — running BERTopic on all starred questions.")
    print("  Party-level analysis will be limited to matched subset.")
    starred["party_family"] = starred["party_family"].fillna("Unknown")
    matched = starred.copy()

# ── 3. Clean question text ─────────────────────────────────────────────────

def clean_text(t):
    if pd.isna(t):
        return ""
    import re
    t = str(t)
    # Remove markdown headers, bullet markers, and boilerplate
    t = re.sub(r"##.*?\n", " ", t)
    t = re.sub(r"\*\d+\.\s+[A-Z\s]+:", " ", t)  # question preamble
    t = re.sub(r"Will the Minister.*?state:", " ", t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r"\-\s*\([a-z]\)", " ", t)        # sub-questions (a) (b) (c)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

matched = matched.copy()
matched["text_clean"] = matched["question_text"].apply(clean_text)
matched = matched[matched["text_clean"].str.len() > 50].reset_index(drop=True)
print(f"  After text cleaning: {len(matched):,} documents")

# ── 4. Fit BERTopic ────────────────────────────────────────────────────────

print("\nLoading sentence transformer...")
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = SentenceTransformer(model_name)

# Custom vectorizer: remove common parliamentary boilerplate
stop_words_custom = [
    "minister", "will", "state", "whether", "government", "please",
    "know", "details", "thereof", "thereon", "thereunder", "said",
    "lok", "sabha", "rajya", "india", "country", "lok sabha",
    "hon", "member", "answer", "question", "reply", "regard",
    "taken", "propose", "considered", "information", "aware",
    "steps", "taken", "measures", "action", "also", "further"
]

vectorizer = CountVectorizer(
    stop_words=stop_words_custom,
    min_df=5,
    max_df=0.85,
    ngram_range=(1, 2)
)

print("Fitting BERTopic (this may take a few minutes)...")
topic_model = BERTopic(
    embedding_model      = embedding_model,
    vectorizer_model     = vectorizer,
    nr_topics            = 30,        # reduce from auto to 30 coherent topics
    min_topic_size       = 15,
    calculate_probabilities = False,  # faster
    verbose              = True
)

docs   = matched["text_clean"].tolist()
topics, _ = topic_model.fit_transform(docs)
matched["topic_id"] = topics
print(f"\nTopics found: {len(set(topics)) - 1} (excluding outlier topic -1)")

# ── 5. Save topic metadata ─────────────────────────────────────────────────

topic_info = topic_model.get_topic_info()
topic_info.to_csv(os.path.join(TABDIR, "bertopic_topics.csv"), index=False)
print("Saved: bertopic_topics.csv")

# Top words per topic (long format)
topic_words = []
for tid in topic_info["Topic"].values:
    if tid == -1:
        continue
    words = topic_model.get_topic(tid)
    for w, score in words[:15]:
        topic_words.append({"topic_id": tid, "word": w, "score": score})

pd.DataFrame(topic_words).to_csv(
    os.path.join(TABDIR, "bertopic_topic_words.csv"), index=False
)
print("Saved: bertopic_topic_words.csv")

# ── 6. Document-level assignments ─────────────────────────────────────────

doc_out = matched[["id", "lok_no", "session_no", "ministry",
                    "party_family", "topic_id"]].copy()
doc_out.to_csv(os.path.join(TABDIR, "bertopic_doc_assignments.csv"), index=False)
print("Saved: bertopic_doc_assignments.csv")

# ── 7. Party × topic prevalence matrix ────────────────────────────────────

# Exclude outlier topic (-1)
filtered = matched[matched["topic_id"] != -1]

party_topic = (
    filtered.groupby(["party_family", "topic_id"])
    .size()
    .reset_index(name="count")
)

# Normalize within party (proportion of questions per topic)
party_totals = filtered.groupby("party_family").size().reset_index(name="total")
party_topic  = party_topic.merge(party_totals, on="party_family")
party_topic["proportion"] = party_topic["count"] / party_topic["total"]

# Pivot to wide matrix
party_matrix = party_topic.pivot_table(
    index="party_family", columns="topic_id",
    values="proportion", fill_value=0
)
party_matrix.to_csv(os.path.join(TABDIR, "bertopic_party_matrix.csv"))
print("Saved: bertopic_party_matrix.csv")

# ── 8. Save model ─────────────────────────────────────────────────────────

model_path = os.path.join(MODDIR, "bertopic_model")
topic_model.save(model_path, serialization="safetensors",
                 save_ctfidf=True, save_embedding_model=model_name)
print(f"Saved model to: {model_path}/")

# ── 9. Summary ────────────────────────────────────────────────────────────

print("\n=== BERTopic Summary ===")
print(topic_info[topic_info["Topic"] != -1][["Topic", "Count", "Name"]].to_string())
n_outlier = (matched["topic_id"] == -1).sum()
print(f"\nOutlier documents (topic -1): {n_outlier:,} ({100*n_outlier/len(matched):.1f}%)")
print("\nB1 complete.")
