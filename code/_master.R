# =============================================================================
# _master.R — Lok Sabha Questions: Unsupervised ML Analysis
# Author: Piyush Zaware
# Last updated: 2026-06-12
#
# PURPOSE:
#   Master script for the Lok Sabha Questions project.
#   Research question: Do Indian parties have coherent ideological identities
#   in parliament, or do MPs cluster by state/region? Has BJP's legislative
#   focus shifted toward Hindutva topics across the 16th–18th Lok Sabha?
#
# DATA:
#   opensansad/lok-sabha-qa (HuggingFace) — 150K starred + unstarred questions
#   16th LS (2014–2019), 17th LS (2019–2024), 18th LS (2024–2026)
#
# PIPELINE:
#   A1 — Download parquet files from HuggingFace
#   A2 — Get MP → party mapping from Parliament API
#   A3 — Merge, filter to starred questions, preprocess text, build DTMs
#   A4 — LDA topic models (K selection via perplexity)
#   A5 — STM with party, Lok Sabha number, ministry covariates
#   A6 — Word2vec embeddings + Kozlowski ideological dimensions
#   A7 — Clustering (HAC + GMM) at party × session level
#   A8 — UMAP / t-SNE + summary figures
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

# -- PATHS -------------------------------------------------------------------

root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
INPDIR <- file.path(root, "input")
CODDIR <- file.path(root, "code")
OUTDIR <- file.path(root, "output")
TMPDIR <- file.path(root, "tmp")

# -- PACKAGES ----------------------------------------------------------------

pkgs <- c(
  "arrow",        # read parquet files
  "httr",         # HTTP requests
  "jsonlite",     # JSON parsing
  "tidyverse",    # wrangling + ggplot2
  "tidytext",     # tidy text mining
  "tm",           # document-term matrix
  "SnowballC",    # stemming
  "topicmodels",  # LDA
  "stm",          # Structural Topic Model
  "word2vec",     # word2vec embeddings
  "mclust",       # GMM clustering
  "umap",         # UMAP
  "Rtsne",        # t-SNE
  "dendextend",   # dendrograms
  "ggrepel",      # plot labels
  "patchwork",    # panel figures
  "kableExtra"    # tables
)

installed  <- rownames(installed.packages())
to_install <- pkgs[!(pkgs %in% installed)]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))

# -- PIPELINE ----------------------------------------------------------------

source(file.path(CODDIR, "A1_download_questions.R"))
source(file.path(CODDIR, "A2_get_party_data.R"))
source(file.path(CODDIR, "A3_preprocess_text.R"))
source(file.path(CODDIR, "A4_lda_topics.R"))
source(file.path(CODDIR, "A5_stm_topics.R"))
source(file.path(CODDIR, "A6_word_embeddings.R"))
source(file.path(CODDIR, "A7_clustering.R"))
source(file.path(CODDIR, "A8_visualization.R"))
source(file.path(CODDIR, "A9_incumbency_robustness.R"))

# -- B PIPELINE: BERTopic + Sentiment + NER (Python → R viz) ----------------
# Independent of A pipeline — runs on individual starred questions
# Requires: pip3 install bertopic sentence-transformers vaderSentiment spacy
#           python3 -m spacy download en_core_web_sm

system(paste("python3", shQuote(file.path(CODDIR, "B1_bertopic.py"))))
source(file.path(CODDIR, "B2_bertopic_viz.R"))
system(paste("python3", shQuote(file.path(CODDIR, "B3_sentiment.py"))))
source(file.path(CODDIR, "B5_sentiment_viz.R"))
system(paste("python3", shQuote(file.path(CODDIR, "B4_ner.py"))))
source(file.path(CODDIR, "B6_ner_viz.R"))

# -- C PIPELINE: Ministry Targeting ------------------------------------------
# Independent of A pipeline — uses raw parquet + party lookup

source(file.path(CODDIR, "C1_ministry_analysis.R"))

cat("\n=== Pipeline complete. Outputs in:", OUTDIR, "===\n")
