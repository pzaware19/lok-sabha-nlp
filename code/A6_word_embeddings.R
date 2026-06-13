# =============================================================================
# A6_word_embeddings.R — Word2Vec + Ideological Dimensions
# Author: Piyush Zaware
# Last updated: 2026-06-12
#
# PURPOSE:
#   Train word2vec on all starred question text (150K+ sentences).
#   Apply Kozlowski et al. (2019) method to recover:
#     - Secular ↔ Hindu Nationalist (Hindutva) dimension
#     - Economic Left ↔ Right dimension
#     - Welfare ↔ Growth dimension
#   Score each party and each Lok Sabha session on these dimensions.
#   Track BJP's Hindutva dimension score across 16th → 17th → 18th LS.
#
# INPUTS:  $INPDIR/questions_with_party.rds
# OUTPUTS:
#   $OUTDIR/models/word2vec.bin
#   $OUTDIR/tables/party_dimensions.csv
#   $OUTDIR/tables/semantic_neighbors.csv
#   $OUTDIR/figures/ideological_space.png
#   $OUTDIR/figures/bjp_hindutva_trajectory.png
# =============================================================================

library(tidyverse)
library(word2vec)
library(SnowballC)
library(ggrepel)
library(tidytext)

set.seed(42)

# ============================================================
# SECTION 1: Prepare training text
# ============================================================
#{

# Train on ALL questions (starred + unstarred) for a richer vocabulary.
# We need rare political terms (hindutva, caste, religion) in the embedding.
questions_all <- readRDS(file.path(INPDIR, "questions_with_party.rds")) %>%
  filter(!is.na(question_text))

questions <- questions_all %>% filter(type == "STARRED")

# Use question_text only (answer texts are very long, making train file >400MB)
# Question text gives enough vocabulary for political/ideological terms
train_text <- questions_all %>%
  mutate(text_clean = coalesce(question_text, "") %>%
    str_to_lower() %>%
    str_remove_all("[^a-z\\s]") %>%
    str_squish()) %>%
  filter(nchar(text_clean) > 20) %>%
  pull(text_clean)

train_file <- file.path(TMPDIR, "w2v_train.txt")
write_lines(train_text, train_file)
cat("Training sentences:", length(train_text), "\n")

#}

# ============================================================
# SECTION 2: Train word2vec
# ============================================================
#{

model_path <- file.path(OUTDIR, "models", "word2vec.bin")
if (file.exists(model_path) &&
    file.mtime(model_path) > file.mtime(train_file)) {
  cat("Loading cached word2vec model...\n")
  w2v <- word2vec::read.word2vec(model_path)
} else {
  cat("Training word2vec (skip-gram, 100 dims)...\n")
  w2v <- word2vec(x = train_file, type = "skip-gram",
                  dim = 100, window = 5, min_count = 5,
                  iter = 10, threads = 4, normalize = TRUE)
  word2vec::write.word2vec(w2v, model_path)
}

E     <- as.matrix(w2v)
vocab <- rownames(E)
cat("Vocabulary:", length(vocab), "words\n")

#}

# ============================================================
# SECTION 3: Define dimension poles
# ============================================================
#{

# Helper: compute dimension vector (Kozlowski method)
# Returns NULL (with warning) if fewer than 2 words found on either pole.
dim_vec <- function(pole_plus, pole_minus, E) {
  p <- pole_plus[pole_plus %in% rownames(E)]
  m <- pole_minus[pole_minus %in% rownames(E)]
  cat("  + pole found:", paste(p, collapse=", "), "\n")
  cat("  - pole found:", paste(m, collapse=", "), "\n")
  if (length(p) < 2 || length(m) < 2) {
    warning("Too few pole words in vocabulary — dimension skipped.")
    return(NULL)
  }
  colMeans(E[p,,drop=FALSE]) - colMeans(E[m,,drop=FALSE])
}

# Project all words onto a dimension (cosine similarity)
project <- function(vec, E) as.numeric(E %*% vec / norm(vec, "2"))

cat("\nDimension 1: Hindutva ↔ Secular\n")
d_hindutva <- dim_vec(
  c("hindu", "temple", "religion", "cow", "mandir", "hindus",
    "ayodhya", "ram", "pilgrimage", "pilgrims"),
  c("secular", "minority", "muslim", "church", "mosque",
    "constitution", "equality", "pluralism"),
  E
)

cat("\nDimension 2: Left ↔ Right (Economic)\n")
d_econ <- dim_vec(
  c("workers", "labour", "union", "wages", "welfare", "subsidy",
    "poor", "redistribution", "public"),
  c("market", "private", "investment", "business", "growth",
    "reform", "capital", "entrepreneur", "investor"),
  E
)

cat("\nDimension 3: Rural/Agrarian ↔ Urban/Digital\n")
d_rural <- dim_vec(
  c("farmer", "agriculture", "crop", "irrigation", "rural",
    "village", "kisan", "drought", "msp"),
  c("urban", "digital", "startup", "technology", "smart",
    "industry", "metro", "infrastructure"),
  E
)

# Project all words — skip any dimension whose vector is NULL
safe_project <- function(vec, E) {
  if (is.null(vec)) return(rep(NA_real_, nrow(E)))
  project(vec, E)
}

word_pos <- tibble(
  word     = vocab,
  hindutva = safe_project(d_hindutva, E),
  econ_l   = safe_project(d_econ,     E),
  rural    = safe_project(d_rural,    E)
)
saveRDS(word_pos, file.path(TMPDIR, "word_positions.rds"))
cat("Dimensions computed. NAs: hindutva=", sum(is.na(word_pos$hindutva)),
    "econ=", sum(is.na(word_pos$econ_l)),
    "rural=", sum(is.na(word_pos$rural)), "\n")

#}

# ============================================================
# SECTION 4: Party × session positions
# ============================================================
#{

tfidf <- readRDS(file.path(TMPDIR, "tfidf_party_session.rds"))
meta  <- read_csv(file.path(TMPDIR, "doc_meta_party.csv"))

party_pos <- tfidf %>%
  left_join(word_pos, by = c("word_stem" = "word")) %>%
  group_by(doc_party_session, party_family, lok_no, session_no) %>%
  summarise(
    pos_hindutva = if (all(is.na(hindutva))) NA_real_ else
                    weighted.mean(hindutva, tf_idf, na.rm=TRUE),
    pos_econ_l   = if (all(is.na(econ_l)))   NA_real_ else
                    weighted.mean(econ_l,   tf_idf, na.rm=TRUE),
    pos_rural    = if (all(is.na(rural)))     NA_real_ else
                    weighted.mean(rural,     tf_idf, na.rm=TRUE),
    .groups = "drop"
  )

write_csv(party_pos, file.path(OUTDIR, "tables", "party_dimensions.csv"))

cat("\nMean Hindutva score by party:\n")
party_pos %>%
  group_by(party_family) %>%
  summarise(hindutva = mean(pos_hindutva)) %>%
  arrange(desc(hindutva)) %>%
  print()

#}

# ============================================================
# SECTION 5: BJP Hindutva trajectory across Lok Sabhas
# ============================================================
#{

bjp_traj <- party_pos %>%
  filter(party_family == "BJP") %>%
  group_by(lok_no) %>%
  summarise(hindutva = mean(pos_hindutva), .groups = "drop") %>%
  mutate(ls_label = paste0(lok_no, "th LS\n(",
                           case_when(lok_no==16~"2014–19",
                                     lok_no==17~"2019–24",
                                     lok_no==18~"2024–26"), ")"))

p_traj <- bjp_traj %>%
  ggplot(aes(lok_no, hindutva)) +
  geom_line(color = "#FF9933", linewidth = 1.8) +
  geom_point(size = 5, color = "#FF9933") +
  geom_text(aes(label = ls_label), vjust = -1.2, size = 3.5) +
  scale_x_continuous(breaks = c(16,17,18)) +
  labs(title = "BJP's Hindutva Score Across Lok Sabhas",
       subtitle = "Based on word2vec dimension (Kozlowski method) from starred questions",
       x = "Lok Sabha", y = "Hindutva Score (higher = more nationalist)") +
  theme_minimal(base_size = 13)

ggsave(file.path(OUTDIR, "figures", "bjp_hindutva_trajectory.png"),
       p_traj, width = 7, height = 5, dpi = 300)

#}

# ============================================================
# SECTION 6: Ideological space plot (party means)
# ============================================================
#{

party_means <- party_pos %>%
  group_by(party_family) %>%
  summarise(hindutva = mean(pos_hindutva),
            econ_l   = mean(pos_econ_l), .groups = "drop")

p_space <- party_means %>%
  ggplot(aes(econ_l, hindutva, label = party_family)) +
  geom_point(size = 4, color = "#636363") +
  geom_text_repel(size = 4, max.overlaps = 15) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey70") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey70") +
  labs(title = "Ideological Space of Indian Parties (Parliamentary Questions)",
       subtitle = "Dimensions recovered via word2vec + Kozlowski method",
       x = "Economic Left ← → Economic Right",
       y = "Secular ← → Hindu Nationalist") +
  theme_minimal(base_size = 13)

ggsave(file.path(OUTDIR, "figures", "ideological_space.png"),
       p_space, width = 9, height = 7, dpi = 300)

#}

# ============================================================
# SECTION 7: Semantic neighbors of key political terms
# ============================================================
#{

get_nn <- function(w, E, n=15) {
  if (!w %in% rownames(E)) return(NULL)
  sims_vec <- (E %*% E[w,])[, 1]  # column matrix → named vector
  sims <- sort(sims_vec, decreasing=TRUE)[2:(n+1)]
  tibble(seed=w, neighbor=names(sims), cosine=as.numeric(sims))
}

seeds <- c("corruption", "farmer", "hindutva", "minority", "development",
           "unemployment", "terrorism", "women", "caste", "inflation")

neighbors <- purrr::map_dfr(seeds, ~get_nn(.x, E))
write_csv(neighbors, file.path(OUTDIR, "tables", "semantic_neighbors.csv"))

p_nn <- neighbors %>%
  group_by(seed) %>% slice_max(cosine, n=8) %>% ungroup() %>%
  mutate(neighbor = reorder_within(neighbor, cosine, seed)) %>%
  ggplot(aes(cosine, neighbor, fill = seed)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~seed, scales = "free_y", ncol = 5) +
  scale_y_reordered() +
  labs(title = "Semantic Neighborhoods in Indian Parliamentary Questions",
       x = "Cosine Similarity", y = NULL) +
  theme_minimal(base_size = 10)

ggsave(file.path(OUTDIR, "figures", "semantic_neighbors.png"),
       p_nn, width = 16, height = 8, dpi = 300)

cat("\nA6 complete.\n")

#}
