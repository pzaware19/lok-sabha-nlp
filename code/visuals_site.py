"""
visuals_site.py
Author: Piyush Zaware
Last updated: 2026-06-18

Generates professional decorative images for the Lok Sabha NLP website:
  1. fig_hero_wordcloud.png   -- full-corpus hero banner (dark navy bg)
  2. fig_ls_era_wordclouds.png -- 3-panel: 16th / 17th / 18th Lok Sabha
  3. fig_party_art.png        -- stylised party vocabulary scatter

IN
  input/questions_raw.csv   (subject + lok_no columns)
  output/figures/ideological_space.png  (already exists)

OUT
  output/figures/fig_hero_wordcloud.png
  output/figures/fig_ls_era_wordclouds.png
  output/figures/fig_party_art.png
"""

import os, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from wordcloud import WordCloud, STOPWORDS

ROOT   = "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
INPDIR = os.path.join(ROOT, "input")
FIGDIR = os.path.join(ROOT, "output", "figures")

# ── Stopwords ─────────────────────────────────────────────────────────────────
PARL_STOPS = {
    "government","india","indian","minister","ministry","state","central",
    "whether","aware","details","steps","taken","will","please","state",
    "information","country","national","provide","year","years","number",
    "total","scheme","schemes","per","also","regard","may","states",
    "made","said","regards","thereof","thereto","reply","stated","suo",
    "motu","regarding","kindly","shri","smt","dr","hon","ble","sir",
    "madam","member","members","question","questions","answer","answers",
    "data","report","list","provided","given","sought","fact","case","cases",
}
ALL_STOPS = STOPWORDS | PARL_STOPS

# ── Load data (subject column only — fast) ────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    os.path.join(INPDIR, "questions_raw.csv"),
    usecols=["lok_no", "subject"],
    dtype={"lok_no": str, "subject": str},
    on_bad_lines="skip",
    nrows=200000,   # 200k subjects is plenty for word clouds
)
df = df.dropna(subset=["subject"])
df["lok_no"] = df["lok_no"].str.strip()
print(f"Loaded {len(df):,} questions")

all_text   = " ".join(df["subject"].tolist())
ls16_text  = " ".join(df[df["lok_no"] == "16"]["subject"].tolist())
ls17_text  = " ".join(df[df["lok_no"] == "17"]["subject"].tolist())
ls18_text  = " ".join(df[df["lok_no"] == "18"]["subject"].tolist())

# ── Colour functions ──────────────────────────────────────────────────────────
def navy_saffron(word, font_size, position, orientation, random_state=None, **kwargs):
    palette = ["#FF9933","#F5E6D3","#FFFFFF","#FFB347","#FFC87C","#E8DDD0","#FF7700"]
    return palette[hash(word) % len(palette)]

def make_era_colour(accent):
    r2, g2, b2 = int(accent[1:3],16), int(accent[3:5],16), int(accent[5:7],16)
    def _c(word, font_size, position, orientation, random_state=None, **kwargs):
        t = (hash(word) % 100) / 100.0
        r = int(0xFF + t*(r2-0xFF)); g = int(0xFF + t*(g2-0xFF)); b = int(0xFF + t*(b2-0xFF))
        return f"#{r:02X}{g:02X}{b:02X}"
    return _c

# ── FIGURE 1: Hero word cloud ─────────────────────────────────────────────────
print("Generating hero word cloud...")
wc = WordCloud(
    width=2400, height=900,
    background_color="#0D1B2A",
    stopwords=ALL_STOPS,
    max_words=350,
    collocations=False,
    color_func=navy_saffron,
    prefer_horizontal=0.82,
    min_font_size=10,
    max_font_size=190,
    relative_scaling=0.45,
    margin=4,
).generate(all_text)

fig, ax = plt.subplots(figsize=(24, 9), facecolor="#0D1B2A")
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
plt.tight_layout(pad=0)
fig.savefig(os.path.join(FIGDIR, "fig_hero_wordcloud.png"),
            dpi=150, bbox_inches="tight", facecolor="#0D1B2A")
plt.close()
print("Saved: fig_hero_wordcloud.png")

# ── FIGURE 2: Era word clouds (3 panels — 16th / 17th / 18th LS) ─────────────
print("Generating era word clouds...")
eras = [
    ("16th Lok Sabha\n(2014–2019)", ls16_text, "#0D1B2A", "#FF9933"),
    ("17th Lok Sabha\n(2019–2024)", ls17_text, "#0F2D18", "#FF9933"),
    ("18th Lok Sabha\n(2024–2026)", ls18_text, "#1A0800", "#F5E6D3"),
]

fig, axes = plt.subplots(1, 3, figsize=(24, 9), facecolor="#111827")
plt.subplots_adjust(hspace=0.03, wspace=0.03)

for ax, (label, text, bg, accent) in zip(axes.flat, eras):
    if not text.strip():
        ax.set_visible(False)
        continue
    wc_era = WordCloud(
        width=780, height=560,
        background_color=bg,
        stopwords=ALL_STOPS,
        max_words=120,
        collocations=False,
        color_func=make_era_colour(accent),
        prefer_horizontal=0.78,
        min_font_size=8,
        max_font_size=110,
        relative_scaling=0.45,
        margin=3,
    ).generate(text)
    ax.imshow(wc_era, interpolation="bilinear")
    ax.axis("off")
    ax.text(0.03, 0.97, label, transform=ax.transAxes,
            color="white", fontsize=13, fontweight="bold", va="top", ha="left",
            path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    # Saffron bottom border on each panel
    ax.axhline(y=wc_era.height - 4, color="#FF9933", linewidth=4, xmin=0, xmax=1)

fig.savefig(os.path.join(FIGDIR, "fig_ls_era_wordclouds.png"),
            dpi=150, bbox_inches="tight", facecolor="#111827")
plt.close()
print("Saved: fig_ls_era_wordclouds.png")

# ── FIGURE 3: Party vocabulary art ───────────────────────────────────────────
print("Generating party art...")

# Use the existing ideological_space data if readable, else make illustrative plot
# Party positions from the ideological_space figure (read off manually)
party_data = {
    "BJP":         ( 0.05,  0.03),
    "INC":         (-0.05, -0.18),
    "AAP":         (-0.40, -0.28),
    "SP":          (-0.35, -0.29),
    "Left":        (-0.22, -0.29),
    "JDU":         ( 0.22,  0.19),
    "RJD":         ( 0.13, -0.09),
    "BSP":         ( 0.22, -0.19),
    "DMK":         ( 0.07, -0.17),
    "TDP":         (-0.28, -0.06),
    "Shiv Sena":   ( 0.06,  0.06),
    "Regional":    (-0.01, -0.21),
    "Independent": ( 0.00,  0.19),
}

party_colours = {
    "BJP":  "#FF9933", "INC": "#1a6bb5", "AAP": "#0066FF",
    "SP":   "#E02020", "Left":"#CC0000", "JDU": "#2E7D32",
    "RJD":  "#27AE60", "BSP": "#4A4A8A", "DMK": "#CC3300",
    "TDP":  "#FFD700", "Shiv Sena": "#FF6600",
    "Regional": "#888888", "Independent": "#AAAAAA",
}

fig, ax = plt.subplots(figsize=(13, 10), facecolor="#0D1B2A")
ax.set_facecolor("#0D1B2A")

# Grid
for v in np.arange(-0.5, 0.6, 0.1):
    ax.axvline(v, color="#152035", linewidth=0.4, zorder=0)
    ax.axhline(v, color="#152035", linewidth=0.4, zorder=0)
ax.axhline(0, color="#2a4a6a", linewidth=1.0, zorder=1)
ax.axvline(0, color="#2a4a6a", linewidth=1.0, zorder=1)

# Quadrant labels
for txt, x, y in [("More state-led", -0.45, 0.28),
                   ("More market", 0.12, 0.28),
                   ("More secular", -0.45, -0.35),
                   ("More nationalist", 0.12, -0.35)]:
    ax.text(x, y, txt, color="#2a4a6a", fontsize=8, alpha=0.7, style="italic")

for party, (px, py) in party_data.items():
    col = party_colours.get(party, "#888888")
    ax.scatter(px, py, color=col, s=220, zorder=4,
               edgecolors="white", linewidths=1.0)
    ax.annotate(party, xy=(px, py),
                xytext=(8, 6), textcoords="offset points",
                color="white", fontsize=9, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#0D1B2A")])

ax.set_xlabel("Economic Left  ←  →  Economic Right",
              color="#cccccc", fontsize=11, labelpad=10)
ax.set_ylabel("Secular  ←  →  Hindu Nationalist",
              color="#cccccc", fontsize=11, labelpad=10)
ax.tick_params(colors="#666")
for spine in ax.spines.values():
    spine.set_edgecolor("#2a4a6a")

ax.set_xlim(-0.52, 0.38)
ax.set_ylim(-0.42, 0.34)

ax.set_title("Indian Party Vocabulary Space (Parliamentary Questions)",
             color="white", fontsize=14, fontweight="bold", pad=14)
ax.text(0.01, -0.07, "Dimensions recovered via word2vec + Kozlowski method",
        transform=ax.transAxes, color="#888", fontsize=8.5, style="italic")

fig.savefig(os.path.join(FIGDIR, "fig_party_art.png"),
            dpi=150, bbox_inches="tight", facecolor="#0D1B2A")
plt.close()
print("Saved: fig_party_art.png")

print("\nvisuals_site.py complete.")
