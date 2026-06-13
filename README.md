# What Does India's Parliament Actually Talk About?
### Unsupervised Machine Learning on 150,000 Lok Sabha Questions (2014–2026)

Every parliamentary session, Indian MPs submit hundreds of written questions to government ministries. Taken together, these questions are a paper trail of what each party actually cares about — agriculture subsidies, minority rights, road infrastructure, defence procurement, labour rights — stated in their own words, on the record.

This project mines the complete archive of starred (oral) questions from the 16th, 17th, and 18th Lok Sabha using unsupervised machine learning: no pre-assigned labels, no ideological coding. Just text, and methods that learn structure from it.

The central question: **can you recover the ideological map of Indian politics purely from how MPs phrase their questions?** The short answer is yes — and the results are sharper than expected.

---

## Data

| Source | HuggingFace: [`opensansad/lok-sabha-qa`](https://huggingface.co/datasets/opensansad/lok-sabha-qa) |
|--------|---|
| Coverage | 16th LS (2014–2019), 17th LS (2019–2024), 18th LS (2024–) |
| Questions | **150,629** total; **11,881** starred (oral) |
| Questions with party | **4,356** starred questions matched to party via Wikipedia |
| Parties covered | BJP, INC, Left (CPI/CPM), BSP, AAP, TMC, SP, JDU, DMK, TDP, RJD, Shiv Sena, Regional |
| MP–party matching | Wikipedia scrape of "List of members of the Nth Lok Sabha" (3 pages) |

All data is downloaded automatically — no manual steps required.

---

## Methodology

```
Raw Questions (150K)
       │
       ▼
  A1: Download        ← HuggingFace parquet (no auth required)
       │
       ▼
  A2: Party Labels    ← Wikipedia scrape (MP → party)
       │
       ▼
  A3: Preprocess      ← Tokenize · Stem · Remove stopwords · Build DTM
       │              ← Documents: party × session (205 docs) + MP-level (199 MPs)
       │
       ├──────────────┬──────────────────┐
       ▼              ▼                  ▼
  A4: LDA         A5: STM          A6: Word2Vec
  (K=15 topics)   (K=20 + party    (Kozlowski 1D
   party heatmap   covariates)      ideological dims)
       │              │                  │
       └──────────────┴──────────────────┘
                      │
                      ▼
              A7: Clustering
              (HAC + GMM on TF-IDF PCA)
                      │
                      ▼
            A8: Visualization
            (UMAP · t-SNE · panels)
```

### Methods at a glance

| Method | What it does | Output |
|--------|-------------|--------|
| **LDA** (Latent Dirichlet Allocation) | Finds 15 recurring topics across all questions | Which topics each party emphasises |
| **STM** (Structural Topic Model) | Like LDA but estimates how party identity and time shift topic usage | BJP vs INC contrasts; 16th→18th LS trends |
| **Word2Vec** (Kozlowski 2019) | Learns word geometry from 150K questions; extracts ideological dimensions from antonym pairs | Each party's position on Hindutva ↔ Secular, Left ↔ Right, Rural ↔ Urban axes |
| **HAC + GMM** | Hierarchical and Gaussian mixture clustering of parties in TF-IDF space | Natural groupings of parties by language |
| **UMAP / t-SNE** | Dimensionality reduction to 2D for visual inspection | Maps where each party-session document sits |

---

## Key Results

### 1. Ideological Space of Indian Parties

Word embeddings recover a recognisable left–right and secular–nationalist structure from parliamentary language alone — with no human annotation.

<p align="center">
  <img src="output/figures/ideological_space.png" width="700"/>
</p>

*Each point is a party. X-axis = economic left↔right; Y-axis = secular↔Hindu nationalist. Positions derived entirely from word co-occurrence patterns in parliamentary questions.*

**Highlights:**
- BJP and Shiv Sena cluster at the Hindu-nationalist end; Left parties and AAP at the secular end
- INC sits in the centre-left — consistent with its "big tent" identity
- The economic axis separates SP/BSP (pro-poor language) from BJP/TDP (market/growth language)

---

### 2. BJP's Hindutva Score Across Three Lok Sabhas

<p align="center">
  <img src="output/figures/bjp_hindutva_trajectory.png" width="600"/>
</p>

*BJP's average position on the Hindutva dimension (higher = more nationalist language) across the 16th, 17th, and 18th Lok Sabha sessions.*

The score increases from the 16th to the 18th LS — the language BJP MPs use in Parliament has shifted measurably toward religious and nationalist vocabulary over a decade in government.

---

### 3. What Each Party Questions — LDA Topic Heatmap

<p align="center">
  <img src="output/figures/lda_party_heatmap.png" width="800"/>
</p>

*Rows = LDA topics (K=15); columns = parties. Colour intensity = average topic weight. Brighter = party strongly emphasises that topic.*

BJP concentrates on infrastructure and defence; INC on accountability and governance; Left parties on labour and wages. Regional parties show sharp, idiosyncratic peaks — NCP on Maharashtra agriculture, DMK on Tamil Nadu water disputes — which makes sense: their mandate is narrow and local.

---

### 4. Document Map: UMAP of Party × Session

<p align="center">
  <img src="output/figures/umap_party_session.png" width="800"/>
</p>

*Each point = one party in one parliamentary session. Plotted in 2D via UMAP from the 3,258-term TF-IDF space. Parties that use similar language cluster together.*

BJP and INC form the two largest, most central clouds. Left parties cluster tightly. Regional parties scatter to the periphery — consistent with their narrow, state-specific focus.

---

### 5. Ministry Focus: Where BJP and INC Spend Their Questions

<p align="center">
  <img src="output/figures/ministry_heatmap.png" width="800"/>
</p>

*Heat = share of questions directed at each ministry, by party. Reveals each party's surveillance priorities.*

---

### 6. Topic Evolution Over Time

<p align="center">
  <img src="output/figures/topic_stacked_bjp.png" width="700"/>
</p>

*BJP topic composition across sessions of the 16th and 17th Lok Sabha. Each band = one LDA topic.*

---

### 7. Semantic Neighbours of Key Political Terms

<p align="center">
  <img src="output/figures/semantic_neighbors.png" width="900"/>
</p>

*Nearest neighbours in the word2vec embedding space for ten politically salient seed words. Reveals how concepts cluster in parliamentary language — e.g. what words appear in the same contexts as "caste", "terrorism", "farmer".*

---

### 8. Hierarchical Clustering of Parties

<p align="center">
  <img src="output/figures/hac_dendrogram.png" width="600"/>
</p>

*Agglomerative clustering on cosine distances between party TF-IDF vectors. The dendrogram reveals which parties use most similar parliamentary language.*

---

## Reproducing the Analysis

**Requirements:** R 4.3+

```r
# Install R and run — everything else is automatic
source("code/_master.R")
```

The master file installs all packages and runs A1 → A8 in sequence. Data is downloaded automatically from HuggingFace (no account needed). Total runtime: ~15 minutes.

### Dependencies

```r
# Data
arrow, httr

# Text
tidytext, tm, SnowballC, topicmodels, stm

# Embeddings
word2vec

# Clustering
mclust, dendextend

# Visualization
ggplot2, ggrepel, umap, Rtsne, patchwork
```

---

## Project Structure

```
Lok_Sabha_Questions/
├── code/
│   ├── _master.R              # Run everything from here
│   ├── A1_download_questions.R
│   ├── A2_get_party_data.R    # Wikipedia scrape → MP-party lookup
│   ├── A3_preprocess_text.R   # Tokenise, stem, build DTM
│   ├── A4_lda_topics.R        # LDA topic model (K=15)
│   ├── A5_stm_topics.R        # Structural Topic Model (K=20)
│   ├── A6_word_embeddings.R   # Word2Vec + Kozlowski dimensions
│   ├── A7_clustering.R        # HAC + GMM clustering
│   └── A8_visualization.R     # UMAP, t-SNE, all figures
├── output/
│   ├── figures/               # 18 publication-quality plots
│   ├── tables/                # CSVs with all numeric results
│   └── models/                # word2vec.bin (keep; large .rds excluded)
└── README.md
```

`input/` and `tmp/` are gitignored (auto-generated by A1).

---

## Theoretical Grounding

This project applies the **Kozlowski, Taddy & Evans (2019)** method (*"The Geometry of Culture"*, American Sociological Review) to Indian parliamentary data. The core idea: ideological dimensions can be encoded as *directions* in a word embedding space, defined by antonym pairs (e.g. `hindu` − `secular`). Party positions are then recovered by projecting their TF-IDF-weighted vocabulary onto these directions.

The approach is validated by the fact that the recovered ordering — Left < AAP < INC < BJP on the Hindutva dimension — matches the widely accepted ideological ordering of Indian parties.

---

## Author

**Piyush Zaware**  
MPP, University of Chicago  
Applied Economics Researcher, Global Poverty Research Laboratory, Northwestern Kellogg
