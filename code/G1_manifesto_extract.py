# =============================================================================
# G1_manifesto_extract.py — Extract text from party election manifestos
# Author: Piyush Zaware
# Updated: 2026-06-14
#
# PURPOSE:
#   Reads PDF manifestos, extracts English text, cleans it, and saves a
#   structured CSV with one row per (party, election_year, lok_no).
#   Maps election year → Lok Sabha number so downstream R code can join
#   directly to the starred-questions parquet data.
#
# NOTES:
#   - Many PDFs have a scanned cover page (0 chars) — start from page 2.
#   - "NON-ASCII" first-page flag in inspection was misleading; actual text
#     content is 95–99% ASCII across all used PDFs.
#   - BJP_2009 / INC_2009 map to 15th LS (not in starred-questions data).
#     They are extracted but flagged lok_no=15 so R can filter them out.
#   - AITC_2019 has very sparse extractable text — excluded.
#   - AIADMK_2009 is Tamil-only — excluded.
#
# INPUTS:  Manifesto/*.pdf
# OUTPUTS: output/tables/manifesto_text.csv
#          output/tables/manifesto_metadata.csv
# =============================================================================

import os, re, csv
import pdfplumber

ROOT      = "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
MAN_DIR   = os.path.join(ROOT, "Manifesto")
OUT_TAB   = os.path.join(ROOT, "output", "tables")
os.makedirs(OUT_TAB, exist_ok=True)

# ── Manifesto registry ────────────────────────────────────────────────────────
# (party_family, election_year, lok_no, filename)
# lok_no = Lok Sabha number elected AT that election
REGISTRY = [
    ("BJP",     2009, 15, "BJP_2009.pdf"),
    ("BJP",     2014, 16, "BJP_2014.pdf"),
    ("BJP",     2019, 17, "BJP_2019.pdf"),
    ("BJP",     2024, 18, "BJP_2024.pdf"),
    ("INC",     2009, 15, "INC_2009.pdf"),
    ("INC",     2014, 16, "INC_2014.pdf"),
    ("INC",     2019, 17, "INC_2019.pdf"),
    ("INC",     2024, 18, "INC_2024.pdf"),
    ("AIADMK",  2014, 16, "AIADMK_Manifesto_2014_85d0f82fcb.pdf"),
    ("AIADMK",  2019, 17, "AIADMK_Manifesto_2019_1abe4f94ae.pdf"),
    ("NCP",     2019, 17, "NCP_2019.pdf"),
    ("NCP",     2024, 18, "NCP_2024.pdf"),
    ("AITC",    2024, 18, "AITC_2924.pdf"),
    ("CPI-M",   2019, 17, "Communist_2019.pdf"),
    ("CPI-M",   2024, 18, "Communist_2024.pdf"),
    ("DMK",     2024, 18, "DMK_Election_Manifesto_Eng_Karthi_33ac415568_ad861eddfe.pdf"),
]

# ── Text cleaning ─────────────────────────────────────────────────────────────
HEADER_RE  = re.compile(r"^(page\s*\d+|www\.\S+|\d+\s*$)", re.IGNORECASE | re.MULTILINE)
BULLETS_RE = re.compile(r"[•■●•]")
MULTI_NL   = re.compile(r"\n{3,}")
NON_PRINT  = re.compile(r"[^\x20-\x7e\n]")

def clean(text: str) -> str:
    text = BULLETS_RE.sub(" ", text)
    text = NON_PRINT.sub(" ", text)          # strip non-printable / Devanagari chars
    text = HEADER_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = MULTI_NL.sub("\n\n", text)
    return text.strip()

def extract_pdf(path: str) -> tuple[str, int, int]:
    """Return (cleaned_text, n_pages_extracted, n_chars)."""
    pages_text = []
    with pdfplumber.open(path) as pdf:
        n_total = len(pdf.pages)
        for pg in pdf.pages:
            raw = pg.extract_text() or ""
            if not raw.strip():
                continue
            # Skip pages that are mostly non-ASCII (Hindi / Tamil cover pages)
            ascii_ratio = sum(1 for c in raw if c.isascii()) / max(len(raw), 1)
            if ascii_ratio < 0.60:
                continue
            pages_text.append(clean(raw))
    combined = "\n\n".join(pages_text)
    return combined, n_total, len(combined)

# ── Extract all ───────────────────────────────────────────────────────────────
rows = []
meta = []
print(f"{'Party':<10} {'Year':<6} {'LS':<4} {'Pages':>6} {'Chars':>8}  File")
print("-" * 70)

for party, year, lok_no, fname in REGISTRY:
    fpath = os.path.join(MAN_DIR, fname)
    if not os.path.exists(fpath):
        print(f"{party:<10} {year:<6} {lok_no:<4} {'MISSING':>6}")
        continue
    try:
        text, n_pages, n_chars = extract_pdf(fpath)
        print(f"{party:<10} {year:<6} {lok_no:<4} {n_pages:>6} {n_chars:>8}  {fname}")
        rows.append({
            "party":         party,
            "election_year": year,
            "lok_no":        lok_no,
            "text":          text,
            "n_chars":       n_chars,
            "source_file":   fname,
        })
        meta.append({
            "party": party, "election_year": year, "lok_no": lok_no,
            "n_chars": n_chars, "n_pdf_pages": n_pages, "source_file": fname,
        })
    except Exception as e:
        print(f"{party:<10} {year:<6} {lok_no:<4} ERROR: {e}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_text = os.path.join(OUT_TAB, "manifesto_text.csv")
out_meta = os.path.join(OUT_TAB, "manifesto_metadata.csv")

with open(out_text, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["party","election_year","lok_no","text","n_chars","source_file"])
    writer.writeheader()
    writer.writerows(rows)

with open(out_meta, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["party","election_year","lok_no","n_chars","n_pdf_pages","source_file"])
    writer.writeheader()
    writer.writerows(meta)

print(f"\nSaved {len(rows)} manifesto documents → {out_text}")
print(f"Metadata → {out_meta}")
