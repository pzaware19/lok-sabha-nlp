"""
B3_sentiment.py — VADER Sentiment Analysis on Lok Sabha Starred Questions
Author: Piyush Zaware
Updated: 2026-06-13

PURPOSE:
    Applies VADER (Valence Aware Dictionary and Sentiment Reasoner) to the
    question text of starred questions. The core hypothesis is that opposition
    parties ask more adversarial questions (negative sentiment) while government
    parties ask softer, aspirational questions about schemes and achievements.

    Dimensions analysed:
    (1) Party-level: mean compound score and adversarialism rate by party
    (2) Ministry-level: which ministries attract the most negative questioning
    (3) Temporal: has question sentiment changed across 16th–18th Lok Sabha?
    (4) Topic-level: BERTopic topic × sentiment (uses bertopic_doc_assignments.csv)
    (5) Starred vs oral supplementary (all questions, not just starred)

    VADER note:
    - Parliamentary English is formal and declarative, not social-media aggressive.
      Compound scores cluster near 0. We care about the *distribution*, not the
      absolute level.
    - Questions like "Has the government failed to..." score negative;
      "What schemes has the government launched..." score positive/neutral.
    - Compound score ∈ [-1, 1]: positive ≥ 0.05, negative ≤ -0.05, else neutral

Inputs:
    tmp/train-*.parquet                        — raw questions (from A1)
    input/mp_party_lookup.csv                  — MP → party_family (1,642 MPs)
    output/tables/bertopic_doc_assignments.csv — topic per question (from B1)

Outputs:
    output/tables/sentiment_doc.csv            — doc-level scores
    output/tables/sentiment_party.csv          — party-level aggregates
    output/tables/sentiment_ministry.csv       — ministry-level aggregates
    output/tables/sentiment_topic.csv          — topic-level aggregates
    output/tables/sentiment_temporal.csv       — lok_no × party aggregates
"""

import os, glob, warnings, re
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT   = "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
TMPDIR = os.path.join(ROOT, "tmp")
TABDIR = os.path.join(ROOT, "output", "tables")
os.makedirs(TABDIR, exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────

print("Loading parquet files...")
parquet_files = sorted(glob.glob(os.path.join(TMPDIR, "train-*.parquet")))
dfs = [pd.read_parquet(f, columns=["id", "lok_no", "session_no", "type",
                                    "ministry", "members", "question_text"])
       for f in parquet_files]
raw = pd.concat(dfs, ignore_index=True)
print(f"  Total rows: {len(raw):,}")

starred = raw[raw["type"] == "STARRED"].copy()
print(f"  Starred questions: {len(starred):,}")

# ── 2. Party matching (same logic as B1) ──────────────────────────────────

lookup = pd.read_csv(os.path.join(ROOT, "input", "mp_party_lookup.csv"))

def norm(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).upper().strip()

def first_member(x):
    try:
        items = list(x)
        return norm(items[0]) if items else ""
    except Exception:
        return norm(str(x)) if x else ""

starred["primary_member"] = starred["members"].apply(first_member)
lookup["mp_key"] = lookup["mp_name"].apply(norm)
mp_to_party = dict(zip(lookup["mp_key"], lookup["party_family"]))
starred["party_family"] = starred["primary_member"].map(mp_to_party)

matched = starred.dropna(subset=["party_family"]).copy()
print(f"  Party-matched: {len(matched):,} ({100*len(matched)/len(starred):.1f}%)")

# ── 3. Clean text for sentiment ────────────────────────────────────────────
# Keep more of the original text — VADER reads raw words, not normalized tokens

def clean_for_sentiment(t):
    if pd.isna(t):
        return ""
    t = str(t)
    # Remove markdown-style headers
    t = re.sub(r"##\s*[A-Z\s]+\n", " ", t)
    # Remove sub-question labels (a) (b)
    t = re.sub(r"\([a-z]\)", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t.strip()

matched["text_sent"] = matched["question_text"].apply(clean_for_sentiment)
matched = matched[matched["text_sent"].str.len() > 20].copy()
print(f"  After text filter: {len(matched):,}")

# ── 4. Run VADER ──────────────────────────────────────────────────────────

print("\nRunning VADER sentiment analysis...")
sia = SentimentIntensityAnalyzer()

def score(text):
    try:
        # VADER on long texts: average paragraph-level scores for stability
        sentences = re.split(r'[.?!]\s+', text)
        sentences = [s for s in sentences if len(s) > 10]
        if not sentences:
            return sia.polarity_scores(text[:512])
        scores = [sia.polarity_scores(s) for s in sentences[:20]]
        return {
            "neg":  np.mean([s["neg"]  for s in scores]),
            "neu":  np.mean([s["neu"]  for s in scores]),
            "pos":  np.mean([s["pos"]  for s in scores]),
            "compound": np.mean([s["compound"] for s in scores])
        }
    except Exception:
        return {"neg": 0, "neu": 1, "pos": 0, "compound": 0}

scores = matched["text_sent"].apply(score)
matched["sent_neg"]      = scores.apply(lambda s: s["neg"])
matched["sent_pos"]      = scores.apply(lambda s: s["pos"])
matched["sent_neu"]      = scores.apply(lambda s: s["neu"])
matched["sent_compound"] = scores.apply(lambda s: s["compound"])

# Classify tone
matched["sent_label"] = matched["sent_compound"].apply(
    lambda c: "adversarial" if c <= -0.05 else ("positive" if c >= 0.05 else "neutral")
)

print(f"  Adversarial questions : {(matched['sent_label']=='adversarial').sum():,} "
      f"({100*(matched['sent_label']=='adversarial').mean():.1f}%)")
print(f"  Positive questions    : {(matched['sent_label']=='positive').sum():,} "
      f"({100*(matched['sent_label']=='positive').mean():.1f}%)")
print(f"  Neutral questions     : {(matched['sent_label']=='neutral').sum():,} "
      f"({100*(matched['sent_label']=='neutral').mean():.1f}%)")

# ── 5. Save document-level output ────────────────────────────────────────

doc_out = matched[["id", "lok_no", "session_no", "party_family", "ministry",
                   "sent_neg", "sent_pos", "sent_neu", "sent_compound",
                   "sent_label"]].copy()
doc_out.to_csv(os.path.join(TABDIR, "sentiment_doc.csv"), index=False)
print("\nSaved: sentiment_doc.csv")

# ── 6. Party-level aggregates ─────────────────────────────────────────────

party_sent = (
    matched.groupby("party_family")
    .agg(
        n_questions     = ("id", "count"),
        mean_compound   = ("sent_compound", "mean"),
        sd_compound     = ("sent_compound", "std"),
        pct_adversarial = ("sent_label", lambda x: 100 * (x == "adversarial").mean()),
        pct_positive    = ("sent_label", lambda x: 100 * (x == "positive").mean()),
        pct_neutral     = ("sent_label", lambda x: 100 * (x == "neutral").mean()),
    )
    .reset_index()
    .sort_values("mean_compound")
)
party_sent.to_csv(os.path.join(TABDIR, "sentiment_party.csv"), index=False)
print("Saved: sentiment_party.csv")
print("\nParty sentiment (most adversarial first):")
print(party_sent[["party_family", "n_questions", "mean_compound",
                   "pct_adversarial"]].to_string(index=False))

# ── 7. Ministry-level aggregates ─────────────────────────────────────────

ministry_sent = (
    matched.groupby("ministry")
    .agg(
        n_questions     = ("id", "count"),
        mean_compound   = ("sent_compound", "mean"),
        pct_adversarial = ("sent_label", lambda x: 100 * (x == "adversarial").mean()),
    )
    .reset_index()
    .query("n_questions >= 30")
    .sort_values("mean_compound")
)
ministry_sent.to_csv(os.path.join(TABDIR, "sentiment_ministry.csv"), index=False)
print("Saved: sentiment_ministry.csv")

# ── 8. Temporal aggregates (lok_no × party) ───────────────────────────────

temporal_sent = (
    matched.groupby(["lok_no", "party_family"])
    .agg(
        n_questions     = ("id", "count"),
        mean_compound   = ("sent_compound", "mean"),
        pct_adversarial = ("sent_label", lambda x: 100 * (x == "adversarial").mean()),
    )
    .reset_index()
)
temporal_sent.to_csv(os.path.join(TABDIR, "sentiment_temporal.csv"), index=False)
print("Saved: sentiment_temporal.csv")

# ── 9. Topic-level sentiment (join with BERTopic assignments) ────────────

bert_path = os.path.join(TABDIR, "bertopic_doc_assignments.csv")
if os.path.exists(bert_path):
    bert_docs = pd.read_csv(bert_path)[["id", "topic_id"]]
    matched_topic = matched.merge(bert_docs, on="id", how="left")

    topic_sent = (
        matched_topic[matched_topic["topic_id"] != -1]
        .groupby("topic_id")
        .agg(
            n_questions     = ("id", "count"),
            mean_compound   = ("sent_compound", "mean"),
            pct_adversarial = ("sent_label", lambda x: 100 * (x == "adversarial").mean()),
        )
        .reset_index()
        .sort_values("mean_compound")
    )
    topic_sent.to_csv(os.path.join(TABDIR, "sentiment_topic.csv"), index=False)
    print("Saved: sentiment_topic.csv")

print("\n=== B3 complete ===")
