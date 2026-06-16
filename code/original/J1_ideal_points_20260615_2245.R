# =============================================================================
# J1_ideal_points.R
# Author: Piyush Zaware
# Updated: 2026-06-14
#
# Goal: Estimate 2D ideal points for MPs from starred question vocabulary
#       using truncated SVD (NOMINATE-style matrix factorization; W3-2 lecture).
#
#       Each MP is a "document", each word is a "dimension".
#       SVD of the TF-IDF matrix gives latent coordinates.
#       Dim 1 = primary axis of vocabulary variation (expected: gov vs opp).
#       Dim 2 = secondary axis (expected: regional vs national).
#
# Inputs:
#   $INPDIR/mp_party_lookup.csv
#   $TMPDIR/train-*.parquet
#
# Outputs:
#   $OUTDIR/figures/ideal_2d.png
#   $OUTDIR/figures/ideal_dim1_party.png
#   $OUTDIR/figures/ideal_dim2_party.png
#   $OUTDIR/figures/ideal_word_loadings.png
#   $OUTDIR/tables/ideal_points.csv
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

pkgs <- c("arrow", "tidyverse", "tidytext", "Matrix", "irlba",
          "ggrepel", "scales", "ggridges")
to_inst <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_inst)) install.packages(to_inst)
suppressPackageStartupMessages(lapply(pkgs, library, character.only = TRUE))

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")

source(file.path(CODDIR, "._stop_words.R"))

MIN_Q    <- 5
MIN_WORD_DOCS <- 5    # word must appear in at least 5 different MPs' corpora

# =============================================================================
# SECTION 1: Load questions and normalize MP names
# =============================================================================
#{
cat("\n[J1] Loading questions from parquet...\n")

strip_hon <- function(s) {
  s <- str_to_upper(str_squish(s))
  str_remove_all(s,
    "\\b(SHRIMATI|SMT\\.?|KUMARI|MRS\\.?|MS\\.?|DR\\.?|PROF\\.?|SH\\.?|SHRI\\.?)\\b")
}
norm_fl <- function(s) {
  parts <- str_split(str_squish(s), "\\s+")[[1]]
  if (length(parts) <= 2) return(s)
  paste(parts[1], parts[length(parts)])
}

parquet_files <- list.files(TMPDIR, pattern = "train-.*\\.parquet$", full.names = TRUE)
raw <- map_dfr(parquet_files, function(f)
  read_parquet(f, col_select = c("lok_no", "type", "members", "question_text")))

starred <- raw %>%
  filter(type == "STARRED", lok_no >= 16L) %>%
  mutate(
    primary_raw  = map_chr(members, function(x)
      tryCatch(str_squish(as.character(list(x)[[1]])[1]), error = function(e) NA_character_)),
    primary_norm = vapply(vapply(replace_na(primary_raw, ""),
                                  strip_hon, character(1)),
                           norm_fl, character(1))
  ) %>%
  filter(primary_norm != "", !is.na(question_text))

cat("  Starred questions loaded:", nrow(starred), "\n")
#}

# =============================================================================
# SECTION 2: Attach party labels from lookup
# =============================================================================
#{
cat("[J1] Attaching party labels...\n")

lookup <- read_csv(file.path(INPDIR, "mp_party_lookup.csv"),
                   show_col_types = FALSE) %>%
  mutate(
    mp_norm = vapply(vapply(str_to_upper(str_squish(mp_name)),
                            strip_hon, character(1)),
                     norm_fl, character(1))
  ) %>%
  arrange(desc(lok_no)) %>%
  distinct(mp_norm, .keep_all = TRUE)

mp_party_map <- setNames(lookup$party_family, lookup$mp_norm)

starred <- starred %>%
  mutate(party_family = mp_party_map[primary_norm]) %>%
  filter(!is.na(party_family))

cat("  After party match:", nrow(starred), "questions from",
    n_distinct(starred$primary_norm), "MPs\n")
#}

# =============================================================================
# SECTION 3: Build MP-level TF-IDF matrix
# =============================================================================
#{
cat("[J1] Building MP-level TF-IDF matrix...\n")

# Filter to MPs with enough questions (pooled across Lok Sabhas)
mp_q_counts <- starred %>%
  count(primary_norm, name = "n_q") %>%
  filter(n_q >= MIN_Q)

tokens <- starred %>%
  semi_join(mp_q_counts, by = "primary_norm") %>%
  unnest_tokens(word, question_text) %>%
  filter(
    !word %in% COMBINED_STOP,
    str_detect(word, "^[a-z]{5,}$")
  )

# Keep words appearing in at least MIN_WORD_DOCS different MPs
word_doc_freq <- tokens %>%
  distinct(primary_norm, word) %>%
  count(word) %>%
  filter(n >= MIN_WORD_DOCS)

tokens_filtered <- tokens %>%
  filter(word %in% word_doc_freq$word)

cat("  Vocabulary:", nrow(word_doc_freq), "words\n")
cat("  MPs in matrix:", n_distinct(tokens_filtered$primary_norm), "\n")

# TF-IDF per MP (pooled)
tfidf_mp <- tokens_filtered %>%
  count(primary_norm, word) %>%
  bind_tf_idf(word, primary_norm, n)
#}

# =============================================================================
# SECTION 4: Build sparse matrix and run truncated SVD
# =============================================================================
#{
cat("[J1] Running truncated SVD...\n")

mp_idx   <- tfidf_mp %>% distinct(primary_norm) %>% mutate(i = row_number())
word_idx <- tfidf_mp %>% distinct(word)          %>% mutate(j = row_number())

mat_df <- tfidf_mp %>%
  left_join(mp_idx,   by = "primary_norm") %>%
  left_join(word_idx, by = "word")

M <- sparseMatrix(
  i    = mat_df$i,
  j    = mat_df$j,
  x    = mat_df$tf_idf,
  dims = c(nrow(mp_idx), nrow(word_idx)),
  dimnames = list(mp_idx$primary_norm, word_idx$word)
)

cat("  Sparse matrix:", nrow(M), "MPs x", ncol(M), "words\n")

# Column-center the matrix: subtract mean TF-IDF per word across all MPs.
# Without centering, Dim 1 captures "typical document frequency" (all parties
# look the same). Centering forces the SVD to explain *deviation* from mean
# usage — the politically meaningful variation.
col_means <- Matrix::colMeans(M)
M_centered <- M - matrix(col_means, nrow = nrow(M), ncol = ncol(M), byrow = TRUE)

svd_res  <- irlba(M_centered, nv = 3)
var_expl <- svd_res$d^2 / sum(svd_res$d^2)

cat(sprintf("  Variance explained: D1=%.1f%%  D2=%.1f%%  D3=%.1f%%\n",
            var_expl[1]*100, var_expl[2]*100, var_expl[3]*100))

mp_coords <- tibble(
  primary_norm = rownames(M),
  dim1 = svd_res$u[, 1] * svd_res$d[1],
  dim2 = svd_res$u[, 2] * svd_res$d[2]
)

word_loadings <- tibble(
  word  = colnames(M),
  load1 = svd_res$v[, 1],
  load2 = svd_res$v[, 2]
)
#}

# =============================================================================
# SECTION 5: Attach party labels to MP coordinates
# =============================================================================
#{
mp_party_tbl <- starred %>%
  distinct(primary_norm, party_family) %>%
  group_by(primary_norm) %>%
  slice(1) %>%
  ungroup()

ideal_pts <- mp_coords %>%
  left_join(mp_party_tbl, by = "primary_norm") %>%
  mutate(party_family = replace_na(party_family, "Other"))

party_colors <- c(
  "BJP"      = "#FF9933",
  "INC"      = "#19AAED",
  "Left"     = "#CC0000",
  "TDP"      = "#DAA520",
  "Shiv Sena"= "#FF6600",
  "JDU"      = "#008000",
  "Regional" = "#9B59B6",
  "Other"    = "#95A5A6"
)

party_centroids <- ideal_pts %>%
  group_by(party_family) %>%
  filter(n() >= 5) %>%
  summarise(dim1 = mean(dim1), dim2 = mean(dim2), n = n(), .groups = "drop")
#}

# =============================================================================
# SECTION 6: Figure 1 — 2D ideal point scatter
# =============================================================================
#{
cat("[J1] Figure 1: 2D ideal point scatter...\n")

p1 <- ggplot(ideal_pts, aes(x = dim1, y = dim2)) +
  geom_point(aes(color = party_family), alpha = 0.30, size = 1.1) +
  geom_point(data = party_centroids, aes(color = party_family),
             size = 5, shape = 18, alpha = 0.95) +
  geom_label_repel(
    data = party_centroids,
    aes(label = paste0(party_family, "\n(n=", n, ")"), color = party_family),
    fontface = "bold", size = 3.2, box.padding = 0.6,
    label.padding = unit(0.15, "lines"), max.overlaps = 15, fill = "white"
  ) +
  scale_color_manual(values = party_colors, na.value = "#95A5A6") +
  labs(
    title = "Parliamentary ideal points from question vocabulary",
    subtitle = sprintf(
      "Each point = one MP. Diamonds = party centroids. SVD on TF-IDF matrix.\nDim 1: %.1f%% variance  |  Dim 2: %.1f%% variance",
      var_expl[1]*100, var_expl[2]*100
    ),
    x = sprintf("Dimension 1 (%.1f%% variance)", var_expl[1]*100),
    y = sprintf("Dimension 2 (%.1f%% variance)", var_expl[2]*100),
    color = NULL,
    caption = "Pooled across 16th-18th Lok Sabha. Stop words, MP names, fragments removed."
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")

ggsave(file.path(FIGDIR, "ideal_2d.png"), p1, width = 10, height = 7, dpi = 150)
file.copy(file.path(FIGDIR, "ideal_2d.png"),
          file.path(dirname(OUTDIR), "docs", "output", "figures", "ideal_2d.png"),
          overwrite = TRUE)
cat("  Saved ideal_2d.png\n")
#}

# =============================================================================
# SECTION 7: Figure 2 — Dim 1 by party (ridge plot)
# =============================================================================
#{
cat("[J1] Figure 2: Dim 1 ridge plot by party...\n")

big_parties <- ideal_pts %>%
  count(party_family) %>%
  filter(n >= 5) %>%
  pull(party_family)

party_order_d1 <- ideal_pts %>%
  filter(party_family %in% big_parties) %>%
  group_by(party_family) %>%
  summarise(med = median(dim1), .groups = "drop") %>%
  arrange(med) %>%
  pull(party_family)

p2 <- ideal_pts %>%
  filter(party_family %in% big_parties) %>%
  mutate(party_family = factor(party_family, levels = party_order_d1)) %>%
  ggplot(aes(x = dim1, y = party_family, fill = party_family)) +
  geom_density_ridges(alpha = 0.7, scale = 0.9, rel_min_height = 0.01) +
  scale_fill_manual(values = party_colors, na.value = "#95A5A6") +
  labs(
    title    = "Dimension 1: primary axis of parliamentary vocabulary",
    subtitle = "Distribution of MP ideal points along Dim 1 by party (sorted by median)",
    x        = sprintf("Dimension 1 (%.1f%% variance)", var_expl[1]*100),
    y        = NULL,
    fill     = NULL,
    caption  = "Sign of dimension is arbitrary: interpret based on party separation."
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")

ggsave(file.path(FIGDIR, "ideal_dim1_party.png"), p2, width = 9, height = 6, dpi = 150)
file.copy(file.path(FIGDIR, "ideal_dim1_party.png"),
          file.path(dirname(OUTDIR), "docs", "output", "figures", "ideal_dim1_party.png"),
          overwrite = TRUE)
cat("  Saved ideal_dim1_party.png\n")
#}

# =============================================================================
# SECTION 8: Figure 3 — Dim 2 by party (ridge plot)
# =============================================================================
#{
cat("[J1] Figure 3: Dim 2 ridge plot by party...\n")

party_order_d2 <- ideal_pts %>%
  filter(party_family %in% big_parties) %>%
  group_by(party_family) %>%
  summarise(med = median(dim2), .groups = "drop") %>%
  arrange(med) %>%
  pull(party_family)

p3 <- ideal_pts %>%
  filter(party_family %in% big_parties) %>%
  mutate(party_family = factor(party_family, levels = party_order_d2)) %>%
  ggplot(aes(x = dim2, y = party_family, fill = party_family)) +
  geom_density_ridges(alpha = 0.7, scale = 0.9, rel_min_height = 0.01) +
  scale_fill_manual(values = party_colors, na.value = "#95A5A6") +
  labs(
    title    = "Dimension 2: secondary axis of parliamentary vocabulary",
    subtitle = "Distribution of MP ideal points along Dim 2 by party (sorted by median)",
    x        = sprintf("Dimension 2 (%.1f%% variance)", var_expl[2]*100),
    y        = NULL,
    fill     = NULL,
    caption  = "Sign of dimension is arbitrary: interpret based on party separation."
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "none")

ggsave(file.path(FIGDIR, "ideal_dim2_party.png"), p3, width = 9, height = 6, dpi = 150)
file.copy(file.path(FIGDIR, "ideal_dim2_party.png"),
          file.path(dirname(OUTDIR), "docs", "output", "figures", "ideal_dim2_party.png"),
          overwrite = TRUE)
cat("  Saved ideal_dim2_party.png\n")
#}

# =============================================================================
# SECTION 9: Figure 4 — Word loadings biplot
# =============================================================================
#{
cat("[J1] Figure 4: word loadings biplot...\n")

top_words <- bind_rows(
  word_loadings %>% arrange(desc(load1)) %>% head(10) %>% mutate(quadrant = "High D1 (+)"),
  word_loadings %>% arrange(load1)       %>% head(10) %>% mutate(quadrant = "Low D1 (-)"),
  word_loadings %>% arrange(desc(load2)) %>% head(10) %>% mutate(quadrant = "High D2 (+)"),
  word_loadings %>% arrange(load2)       %>% head(10) %>% mutate(quadrant = "Low D2 (-)")
) %>%
  distinct(word, .keep_all = TRUE)

quad_colors <- c(
  "High D1 (+)" = "#e66101", "Low D1 (-)" = "#5e3c99",
  "High D2 (+)" = "#1a9641", "Low D2 (-)" = "#d7191c"
)

p4 <- ggplot(top_words, aes(x = load1, y = load2, label = word, color = quadrant)) +
  geom_point(alpha = 0.7, size = 2.5) +
  geom_text_repel(size = 3.5, max.overlaps = 25, fontface = "bold",
                  min.segment.length = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey60") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
  scale_color_manual(values = quad_colors) +
  labs(
    title    = "What words define each dimension?",
    subtitle = "Top 10 words by loading in each quadrant of the 2D SVD space",
    x        = "Dimension 1 loading",
    y        = "Dimension 2 loading",
    color    = NULL,
    caption  = "Words closer to an axis pole pull MPs in that direction in the ideal point space."
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")

ggsave(file.path(FIGDIR, "ideal_word_loadings.png"), p4, width = 9, height = 7, dpi = 150)
file.copy(file.path(FIGDIR, "ideal_word_loadings.png"),
          file.path(dirname(OUTDIR), "docs", "output", "figures", "ideal_word_loadings.png"),
          overwrite = TRUE)
cat("  Saved ideal_word_loadings.png\n")
#}

# =============================================================================
# SECTION 10: Save ideal points table
# =============================================================================
#{
write_csv(
  ideal_pts %>% select(primary_norm, party_family, dim1, dim2),
  file.path(TABDIR, "ideal_points.csv")
)
cat("[J1] Done. Saved ideal_points.csv with", nrow(ideal_pts), "MPs.\n\n")

# Print party centroid summary
cat("Party centroids:\n")
print(party_centroids %>% arrange(dim1))
#}
