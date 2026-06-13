# =============================================================================
# A3_preprocess_text.R — Text Preprocessing
# Author: Piyush Zaware
# Last updated: 2026-06-12
#
# PURPOSE:
#   Preprocess Lok Sabha question text.
#   Build two document-level corpora:
#     (a) Party × Session — one document per party per Lok Sabha session
#         (for topic models with party/time covariates)
#     (b) MP-level — one document per MP aggregated across all questions
#         (for ideal-point estimation and MP-level clustering)
#   Filter to STARRED questions only (oral, richer text).
#
# INPUTS:  $INPDIR/questions_with_party.rds
# OUTPUTS:
#   $TMPDIR/dtm_party_session.rds   — DTM: party × session
#   $TMPDIR/dtm_mp.rds              — DTM: MP level
#   $TMPDIR/tfidf_party_session.rds — TF-IDF: party × session
#   $TMPDIR/tfidf_wide.rds          — wide TF-IDF matrix for clustering
#   $TMPDIR/doc_meta_party.csv      — metadata for party × session docs
#   $TMPDIR/doc_meta_mp.csv         — metadata for MP docs
#   $TMPDIR/vocab.csv               — final vocabulary
# =============================================================================

library(tidyverse)
library(tidytext)
library(tm)
library(SnowballC)

set.seed(42)

# ============================================================
# SECTION 1: Load and filter
# ============================================================
#{

questions <- readRDS(file.path(INPDIR, "questions_with_party.rds"))

# Filter to starred questions with known party
starred <- questions %>%
  filter(
    type == "STARRED",
    party_family != "Unknown",
    !is.na(question_text),
    nchar(question_text) > 50
  )

cat("Starred questions with party:", nrow(starred), "\n")
cat("Lok Sabha coverage:", paste(sort(unique(starred$lok_no)), collapse=", "), "\n")
cat("Parties:", paste(sort(unique(starred$party_family)), collapse=", "), "\n")

#}

# ============================================================
# SECTION 2: Define stopwords
# ============================================================
#{

parliament_stopwords <- c(
  stopwords("english"),
  # parliamentary boilerplate
  "whether", "minister", "government", "state", "will", "may",
  "please", "stated", "inform", "house", "consider", "taken",
  "steps", "regard", "details", "thereof", "therein", "also",
  "india", "country", "national", "central", "scheme", "year",
  "number", "total", "made", "given", "said", "per", "time",
  "can", "reply", "question", "answer", "member", "parliament",
  # Hindi transliterations that survive into English text
  "ke", "ki", "ka", "hai", "ko", "se", "ka", "aur", "yojana"
)

#}

# ============================================================
# SECTION 3: Tokenize
# ============================================================
#{

# Create document ID: party_family × lok_no × session_no
starred <- starred %>%
  mutate(
    doc_party_session = paste(party_family, lok_no, session_no, sep = "_"),
    doc_mp            = primary_mp_upper
  )

tokens <- starred %>%
  select(doc_party_session, doc_mp, party_family, lok_no,
         session_no, date_parsed, question_text) %>%
  unnest_tokens(word, question_text) %>%
  filter(!word %in% parliament_stopwords) %>%
  filter(!str_detect(word, "^[0-9]+$")) %>%
  filter(nchar(word) > 2) %>%
  mutate(word_stem = wordStem(word, language = "english")) %>%
  filter(nchar(word_stem) > 2)

cat("Tokens after cleaning:", nrow(tokens), "\n")

#}

# ============================================================
# SECTION 4: Vocabulary pruning
# ============================================================
#{

# Keep terms appearing in ≥ 5 documents
n_docs     <- n_distinct(tokens$doc_party_session)
term_docs  <- tokens %>%
  distinct(doc_party_session, word_stem) %>%
  dplyr::count(word_stem)

vocab <- term_docs %>%
  filter(n >= 5, n <= n_docs * 0.90) %>%
  pull(word_stem)

cat("Vocabulary size:", length(vocab), "\n")

tokens_clean <- tokens %>% filter(word_stem %in% vocab)
write_csv(tibble(word_stem = vocab), file.path(TMPDIR, "vocab.csv"))

#}

# ============================================================
# SECTION 5: Party × Session DTM
# ============================================================
#{

party_session_counts <- tokens_clean %>%
  dplyr::count(doc_party_session, party_family, lok_no, session_no, word_stem)

dtm_party_session <- party_session_counts %>%
  cast_dtm(document = doc_party_session, term = word_stem, value = n)

dtm_party_session <- dtm_party_session[rowSums(as.matrix(dtm_party_session)) > 0, ]
cat("Party × Session DTM:", dim(dtm_party_session), "\n")
saveRDS(dtm_party_session, file.path(TMPDIR, "dtm_party_session.rds"))

# Metadata
doc_meta_party <- party_session_counts %>%
  distinct(doc_party_session, party_family, lok_no, session_no) %>%
  mutate(year = case_when(
    lok_no == 16 ~ 2014L + as.integer(session_no %/% 3),
    lok_no == 17 ~ 2019L + as.integer(session_no %/% 3),
    lok_no == 18 ~ 2024L,
    TRUE ~ NA_integer_
  ))
write_csv(doc_meta_party, file.path(TMPDIR, "doc_meta_party.csv"))

#}

# ============================================================
# SECTION 6: TF-IDF (party × session)
# ============================================================
#{

tfidf_party <- party_session_counts %>%
  bind_tf_idf(term = word_stem, document = doc_party_session, n = n) %>%
  left_join(doc_meta_party, by = c("doc_party_session", "party_family",
                                    "lok_no", "session_no"))

saveRDS(tfidf_party, file.path(TMPDIR, "tfidf_party_session.rds"))

# Wide matrix for PCA / clustering
tfidf_wide <- tfidf_party %>%
  select(doc_party_session, word_stem, tf_idf) %>%
  pivot_wider(names_from = word_stem, values_from = tf_idf, values_fill = 0)

saveRDS(tfidf_wide, file.path(TMPDIR, "tfidf_wide.rds"))

#}

# ============================================================
# SECTION 7: MP-level DTM (for ideal point estimation)
# ============================================================
#{

# Filter MPs with ≥ 10 questions for reliable estimation
active_mps <- starred %>%
  dplyr::count(doc_mp, party_family) %>%
  filter(n >= 10) %>%
  pull(doc_mp)

cat("MPs with ≥10 questions:", length(active_mps), "\n")

mp_counts <- tokens_clean %>%
  filter(doc_mp %in% active_mps) %>%
  dplyr::count(doc_mp, word_stem)

dtm_mp <- mp_counts %>%
  cast_dtm(document = doc_mp, term = word_stem, value = n)

dtm_mp <- dtm_mp[rowSums(as.matrix(dtm_mp)) > 0, ]
cat("MP-level DTM:", dim(dtm_mp), "\n")
saveRDS(dtm_mp, file.path(TMPDIR, "dtm_mp.rds"))

doc_meta_mp <- starred %>%
  filter(doc_mp %in% active_mps) %>%
  distinct(doc_mp, primary_mp, party_family, lok_no) %>%
  group_by(doc_mp) %>%
  slice(1) %>%
  ungroup()
write_csv(doc_meta_mp, file.path(TMPDIR, "doc_meta_mp.csv"))

cat("\nA3 complete.\n")

#}
