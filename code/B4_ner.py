"""
B4_ner.py — Named Entity Recognition on Lok Sabha Starred Questions
Author: Piyush Zaware
Updated: 2026-06-13

PURPOSE:
    Applies spaCy NER to extract:
    (1) GPE (Geo-Political Entity) — states, districts, countries mentioned
        → which parties advocate for which geographic constituencies?
    (2) PERSON — individuals named (politicians, bureaucrats, activists)
        → are parties asking about specific individuals more than others?
    (3) ORG — organisations and institutions mentioned

    Parliamentary English uses formal, repetitive language; many questions
    name states in their opening ("the state of Maharashtra", "districts in UP").
    The geographic pattern is the most politically interpretable: regional parties
    should cluster on their home state; national parties should be more diffuse.

    Output metrics per party-entity pair:
    - Raw mention count
    - Questions mentioning the entity (distinct question count)
    - Share of party's total questions mentioning the entity

Inputs:
    tmp/train-*.parquet                — raw questions (from A1)
    input/mp_party_lookup.csv          — MP → party_family (1,642 MPs)

Outputs:
    output/tables/ner_gpe_party.csv    — GPE × party mention matrix
    output/tables/ner_person_party.csv — PERSON × party mention matrix
    output/tables/ner_org_party.csv    — ORG × party mention matrix
    output/tables/ner_doc.csv          — doc-level entity counts
"""

import os, glob, warnings, re
import pandas as pd
import numpy as np
import spacy
from collections import Counter
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT   = "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
TMPDIR = os.path.join(ROOT, "tmp")
TABDIR = os.path.join(ROOT, "output", "tables")
os.makedirs(TABDIR, exist_ok=True)

# ── 1. Load spaCy model ────────────────────────────────────────────────────

print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("  en_core_web_sm not found. Run: python3 -m spacy download en_core_web_sm")
    raise

# Disable unnecessary pipeline components for speed
nlp.disable_pipes([p for p in nlp.pipe_names if p not in ("tok2vec", "ner")])
print(f"  spaCy version: {spacy.__version__}")

# ── 2. Load data and party-match ──────────────────────────────────────────

print("\nLoading parquet files...")
parquet_files = sorted(glob.glob(os.path.join(TMPDIR, "train-*.parquet")))
dfs = [pd.read_parquet(f, columns=["id", "lok_no", "session_no", "type",
                                    "ministry", "members", "question_text"])
       for f in parquet_files]
raw = pd.concat(dfs, ignore_index=True)

starred = raw[raw["type"] == "STARRED"].copy()
print(f"  Starred questions: {len(starred):,}")

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

# ── 3. Clean text for NER ─────────────────────────────────────────────────

INDIA_STATES = {
    "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH",
    "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JHARKHAND", "KARNATAKA",
    "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA",
    "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM",
    "TAMIL NADU", "TELANGANA", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND",
    "WEST BENGAL", "DELHI", "JAMMU AND KASHMIR", "LADAKH", "PUDUCHERRY",
    "ANDAMAN AND NICOBAR", "LAKSHADWEEP", "CHANDIGARH",
    "UP", "MP", "AP", "HP", "J&K"
}

# Non-person tokens that spaCy mis-tags as PERSON
NON_PERSON_TOKENS = INDIA_STATES | {
    # Honorifics alone (without a name)
    "SHRIMATI", "SHRI", "DR", "PROF", "SMT", "SH",
    # Scheme / programme names
    "PMAY", "PMGSY", "MGNREGS", "NREGA", "AYUSHMAN", "PRADHAN",
    "JALDOOT", "SWACHH", "UJJWALA", "KISAN", "KISHAN",
    # Common non-person nouns in parliament
    "MINISTER", "PRIME MINISTER", "PRESIDENT", "SPEAKER", "CHAIRMAN",
    "SECRETARY", "HONORABLE", "HON", "MEMBER", "SIR", "MADAM",
    "THE PRIME MINISTER", "THE MINISTER", "RAILWAYS", "STATEWISE",
    "CONSTITUENCY", "QUESTION", "ANSWER", "GOVERNMENT", "BACKWARD CLASSES",
    "GRAM PANCHAYATS", "COVID", "DEAN", "PATEL", "GEORGE BAKER",
    "QUESTION NO", "DISTRIBUTION LOK SABHA STARRED",
}

_VOWELS = set("AEIOU")
_NON_PERSON_WORDS = {
    "STATEWISE", "RAILWAYS", "DISTRIBUTION", "BACKWARD", "GRAM", "COVID",
    "CONSTITUENCY", "QUESTION", "PANCHAYATS", "COAL", "KENDRAS", "TRIBES",
    "JHANSI", "KOTAGIRI", "VKSJ", "BILL", "COVID-19", "COVID19",
    "SCHEDULED", "BHAGALPUR", "SEONI", "SITAMARHI", "KENDRIYA",
    "VIDYALAYAS", "MAHATMA",    # Mahatma is a title, not a surname alone
}

def _is_valid_person(text_u):
    """Return True if the entity looks like a real person name."""
    # Devanagari / Hindi characters → garbled transliteration
    if re.search(r'[ऀ-ॿऀ-ॿ]', text_u):
        return False
    # Strip to letters only for further checks
    clean = re.sub(r'[^A-Z\s]', '', text_u).strip()
    if len(clean) < 4:
        return False
    # All tokens ≤ 3 chars (initials only, no real name component)
    tokens = clean.split()
    if tokens and all(len(t) <= 3 for t in tokens):
        return False
    # Any single token is >4 chars but has <25% vowels → garbled transliteration
    for t in tokens:
        if len(t) > 4:
            vowel_ratio = sum(1 for c in t if c in _VOWELS) / len(t)
            if vowel_ratio < 0.15:   # e.g. VKSJ=0%, LKOZTFUD=12.5%
                return False
    # Exact match in blocklist
    if text_u in NON_PERSON_TOKENS:
        return False
    # Any token matches a known non-person word
    for tok in text_u.split():
        if tok in INDIA_STATES or tok in _NON_PERSON_WORDS:
            return False
    return True

def _clean_person_name(text_u):
    """Strip leading honorific prefixes for cleaner display."""
    for prefix in ("SHRIMATI ", "SHRI ", "DR. ", "DR ", "PROF. ", "PROF ", "SMT. ", "SMT "):
        if text_u.startswith(prefix):
            text_u = text_u[len(prefix):]
    # Remove trailing colon or punctuation
    return text_u.rstrip(":.,;")

def clean_text(t):
    if pd.isna(t):
        return ""
    t = str(t)
    t = re.sub(r"##\s*[A-Z\s]+\n", " ", t)
    t = re.sub(r"\([a-z]\)", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()[:2000]  # cap at 2000 chars for speed

matched["text_ner"] = matched["question_text"].apply(clean_text)
matched = matched[matched["text_ner"].str.len() > 20].reset_index(drop=True)
print(f"  After text filter: {len(matched):,}")

# ── 4. Run NER in batches ─────────────────────────────────────────────────

print(f"\nRunning spaCy NER on {len(matched):,} documents...")
print("  (this takes ~3–5 minutes)")

# Collect per-document entity lists
gpe_lists    = []
person_lists = []
org_lists    = []

batch_size = 500
texts = matched["text_ner"].tolist()
doc_ids = matched["id"].tolist()

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    docs = list(nlp.pipe(batch, batch_size=64))
    for doc in docs:
        gpe    = []
        person = []
        org    = []
        for ent in doc.ents:
            text_u = ent.text.strip().upper()
            if ent.label_ == "GPE":
                gpe.append(text_u)
            elif ent.label_ == "PERSON":
                if _is_valid_person(text_u):
                    person.append(_clean_person_name(text_u))
            elif ent.label_ == "ORG":
                if len(text_u) > 2:
                    org.append(text_u)
        gpe_lists.append(gpe)
        person_lists.append(person)
        org_lists.append(org)

    if (i // batch_size + 1) % 5 == 0:
        print(f"  Processed {min(i+batch_size, len(texts)):,}/{len(texts):,}")

matched["gpe_ents"]    = gpe_lists
matched["person_ents"] = person_lists
matched["org_ents"]    = org_lists

print("  NER complete.")

# ── 5. Save document-level entity counts ─────────────────────────────────

doc_out = matched[["id", "lok_no", "session_no", "party_family", "ministry"]].copy()
doc_out["n_gpe"]    = [len(g) for g in gpe_lists]
doc_out["n_person"] = [len(p) for p in person_lists]
doc_out["n_org"]    = [len(o) for o in org_lists]
# State detection: combine spaCy GPE tags with direct string matching.
# spaCy en_core_web_sm sometimes tags "Tamil Nadu" as NORP instead of GPE,
# so direct matching is more reliable for Indian state names.
import re as _re

STATE_PATTERNS = {s: _re.compile(r'\b' + _re.escape(s.title()) + r'\b', _re.IGNORECASE)
                  for s in INDIA_STATES}
# Also match common abbreviations
STATE_PATTERNS["UP"] = _re.compile(r'\bUP\b')
STATE_PATTERNS["MP"] = _re.compile(r'\bM\.?P\.?\b')

def extract_states(text, gpe_list):
    """Merge NER GPE states with direct regex matching."""
    found = set()
    for g in gpe_list:
        if g in INDIA_STATES:
            found.add(g)
    # Direct string match — catches what NER misses (e.g. Tamil Nadu tagged as NORP)
    for state, pat in STATE_PATTERNS.items():
        if pat.search(text):
            found.add(state)
    return found

doc_out["states_mentioned"] = [
    ";".join(sorted(extract_states(txt, gpe)))
    for txt, gpe in zip(matched["text_ner"].tolist(), gpe_lists)
]
doc_out.to_csv(os.path.join(TABDIR, "ner_doc.csv"), index=False)
print("\nSaved: ner_doc.csv")

# ── 6. Build party × entity matrices ─────────────────────────────────────

def build_party_entity_table(matched, ent_col, label, top_n=50):
    """
    For each party, count distinct questions mentioning each entity.
    Returns long-format table: party_family, entity, n_questions, share_of_party
    """
    rows = []
    for _, row in matched.iterrows():
        for ent in set(row[ent_col]):  # set → count each entity once per question
            rows.append({"party_family": row["party_family"],
                         "entity": ent,
                         "question_id": row["id"]})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    counts = (df.groupby(["party_family", "entity"])
                .size()
                .reset_index(name="n_questions"))

    party_totals = matched.groupby("party_family")["id"].count().reset_index(name="party_total")
    counts = counts.merge(party_totals, on="party_family")
    counts["share_pct"] = 100 * counts["n_questions"] / counts["party_total"]

    # Keep top_n entities by total mention count
    top_entities = (counts.groupby("entity")["n_questions"]
                         .sum()
                         .nlargest(top_n)
                         .index.tolist())
    counts = counts[counts["entity"].isin(top_entities)]

    return counts.sort_values(["party_family", "n_questions"], ascending=[True, False])

print("\nBuilding GPE × party table...")
gpe_party = build_party_entity_table(matched, "gpe_ents", "GPE", top_n=60)
gpe_party.to_csv(os.path.join(TABDIR, "ner_gpe_party.csv"), index=False)
print(f"Saved: ner_gpe_party.csv ({len(gpe_party):,} rows)")

print("Building PERSON × party table...")
person_party = build_party_entity_table(matched, "person_ents", "PERSON", top_n=50)
person_party.to_csv(os.path.join(TABDIR, "ner_person_party.csv"), index=False)
print(f"Saved: ner_person_party.csv ({len(person_party):,} rows)")

print("Building ORG × party table...")
org_party = build_party_entity_table(matched, "org_ents", "ORG", top_n=60)
org_party.to_csv(os.path.join(TABDIR, "ner_org_party.csv"), index=False)
print(f"Saved: ner_org_party.csv ({len(org_party):,} rows)")

# ── 7. Summary stats ─────────────────────────────────────────────────────

print("\n=== NER Summary ===")
print(f"Avg GPE entities per question:    {doc_out['n_gpe'].mean():.2f}")
print(f"Avg PERSON entities per question: {doc_out['n_person'].mean():.2f}")
print(f"Avg ORG entities per question:    {doc_out['n_org'].mean():.2f}")

# Top 15 Indian states by mention
state_counts = Counter()
for gpe_list in matched["gpe_ents"]:
    for g in gpe_list:
        if g in INDIA_STATES:
            state_counts[g] += 1
print("\nTop 15 Indian states mentioned:")
for state, count in state_counts.most_common(15):
    print(f"  {state:<30} {count:>5}")

# State mentions per party
print("\nState mentions per party (top 5 states each):")
for party in sorted(matched["party_family"].unique()):
    sub = matched[matched["party_family"] == party]
    sc = Counter()
    for gpe_list in sub["gpe_ents"]:
        for g in gpe_list:
            if g in INDIA_STATES:
                sc[g] += 1
    if sc:
        top5 = ", ".join([f"{k}({v})" for k, v in sc.most_common(5)])
        print(f"  {party:<12}: {top5}")

print("\n=== B4 complete ===")
