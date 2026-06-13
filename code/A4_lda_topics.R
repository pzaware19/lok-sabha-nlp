# =============================================================================
# A4_lda_topics.R — LDA Topic Models
# Author: Piyush Zaware
# Last updated: 2026-06-12
#
# PURPOSE:
#   Fit LDA on party × session documents.
#   Select K via held-out perplexity. Interpret topics.
#   Compare topic usage by party (BJP vs INC vs Left vs Regional).
#
# INPUTS:  $TMPDIR/dtm_party_session.rds, doc_meta_party.csv
# OUTPUTS:
#   $OUTDIR/models/lda_k{K}.rds
#   $OUTDIR/tables/lda_top_terms.csv
#   $OUTDIR/tables/lda_party_topics.csv
#   $OUTDIR/figures/lda_k_selection.png
#   $OUTDIR/figures/lda_top_terms.png
#   $OUTDIR/figures/lda_party_heatmap.png
# =============================================================================

library(tidyverse)
library(tidytext)
library(topicmodels)

set.seed(42)

# ============================================================
# SECTION 1: Load
# ============================================================
#{

dtm      <- readRDS(file.path(TMPDIR, "dtm_party_session.rds"))
doc_meta <- read_csv(file.path(TMPDIR, "doc_meta_party.csv"))

dtm <- dtm[rowSums(as.matrix(dtm)) > 0, ]
cat("DTM:", dim(dtm), "\n")

#}

# ============================================================
# SECTION 2: K selection via held-out perplexity
# ============================================================
#{

K_candidates <- c(8, 10, 15, 20, 25, 30)

n      <- nrow(dtm)
train  <- sample(1:n, floor(0.8 * n))
dtm_tr <- dtm[train, ]
dtm_te <- dtm[-train, ]

# Keep only terms present in training set
tr_terms <- names(which(colSums(as.matrix(dtm_tr)) > 0))
dtm_tr   <- dtm_tr[, tr_terms]
dtm_te   <- dtm_te[, tr_terms]

perp_results <- purrr::map_dfr(K_candidates, function(k) {
  cat("LDA K =", k, "...")
  m <- LDA(dtm_tr, k = k, method = "Gibbs",
           control = list(seed = 42, iter = 1000, burnin = 200, thin = 10))
  p <- perplexity(m, dtm_te)
  cat(" perplexity =", round(p, 1), "\n")
  tibble(K = k, perplexity = p)
})

write_csv(perp_results, file.path(OUTDIR, "tables", "lda_perplexity.csv"))

p_perp <- ggplot(perp_results, aes(K, perplexity)) +
  geom_line(color = "#2c7bb6", linewidth = 1) +
  geom_point(size = 3, color = "#2c7bb6") +
  scale_x_continuous(breaks = K_candidates) +
  labs(title = "LDA K Selection: Held-out Perplexity",
       x = "Number of Topics (K)", y = "Perplexity") +
  theme_minimal(base_size = 13)

ggsave(file.path(OUTDIR, "figures", "lda_k_selection.png"),
       p_perp, width = 7, height = 4, dpi = 300)

K_final <- perp_results %>%
  mutate(delta = c(NA, diff(perplexity))) %>%
  filter(!is.na(delta)) %>%
  slice_min(delta) %>%
  pull(K)
cat("Selected K:", K_final, "\n")

#}

# ============================================================
# SECTION 3: Final model
# ============================================================
#{

lda_final <- LDA(dtm, k = K_final, method = "Gibbs",
                 control = list(seed = 42, iter = 2000, burnin = 500, thin = 10))
saveRDS(lda_final, file.path(OUTDIR, "models", paste0("lda_k", K_final, ".rds")))

#}

# ============================================================
# SECTION 4: Top terms per topic
# ============================================================
#{

top_terms <- tidy(lda_final, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 15) %>%
  ungroup()

write_csv(top_terms, file.path(OUTDIR, "tables", "lda_top_terms.csv"))

cat("\n=== LDA TOPICS — label these manually in A5 ===\n")
top_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 8) %>%
  summarise(terms = paste(term, collapse = ", ")) %>%
  print(n = Inf)

p_terms <- top_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = 8) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~topic, scales = "free_y", ncol = 5) +
  scale_y_reordered() +
  labs(title = paste0("LDA Topics (K=", K_final, ")"),
       x = "β (word-topic probability)", y = NULL) +
  theme_minimal(base_size = 9)

ggsave(file.path(OUTDIR, "figures", "lda_top_terms.png"),
       p_terms, width = 16, height = 10, dpi = 300)

#}

# ============================================================
# SECTION 5: Party × topic heatmap
# ============================================================
#{

lda_gamma <- tidy(lda_final, matrix = "gamma") %>%
  rename(doc_party_session = document) %>%
  left_join(doc_meta, by = "doc_party_session")

write_csv(lda_gamma, file.path(OUTDIR, "tables", "lda_party_topics.csv"))

party_avg <- lda_gamma %>%
  group_by(party_family, topic) %>%
  summarise(mean_gamma = mean(gamma), .groups = "drop")

p_heat <- party_avg %>%
  ggplot(aes(factor(topic), party_family, fill = mean_gamma)) +
  geom_tile(color = "white") +
  scale_fill_distiller(palette = "YlOrRd", direction = 1, name = "Topic\nShare") +
  labs(title = "Party × Topic Distribution (LDA)",
       subtitle = "Lok Sabha starred questions, 16th–18th LS",
       x = "Topic", y = NULL) +
  theme_minimal(base_size = 12)

ggsave(file.path(OUTDIR, "figures", "lda_party_heatmap.png"),
       p_heat, width = 12, height = 6, dpi = 300)

cat("\nA4 complete. K =", K_final, "\n")

#}
