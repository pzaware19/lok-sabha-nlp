# =============================================================================
# A5_stm_topics.R — Structural Topic Model
# Author: Piyush Zaware
# Last updated: 2026-06-12
#
# PURPOSE:
#   Fit STM with covariates: party family, Lok Sabha number, session number.
#   Key questions:
#     - Which topics are BJP-specific vs. INC-specific?
#     - Has BJP's topic mix shifted from 16th to 17th to 18th LS?
#     - Do Left parties dominate labour/welfare topics?
#
# INPUTS:  $TMPDIR/dtm_party_session.rds, doc_meta_party.csv
# OUTPUTS:
#   $OUTDIR/models/stm_k{K}.rds
#   $OUTDIR/tables/stm_top_terms.csv
#   $OUTDIR/tables/stm_covariate_effects.csv
#   $OUTDIR/figures/stm_coherence_frontier.png
#   $OUTDIR/figures/stm_effects_lok.png
#   $OUTDIR/figures/stm_bjp_inc_contrast.png
# =============================================================================

library(tidyverse)
library(stm)
library(ggrepel)

set.seed(42)

# ============================================================
# SECTION 1: Load and prepare
# ============================================================
#{

dtm <- readRDS(file.path(TMPDIR, "dtm_party_session.rds"))

# Use dtm$dimnames$Docs for robust document ID extraction
doc_ids <- dtm$dimnames$Docs
cat("DTM dimensions:", nrow(dtm), "x", ncol(dtm), "\n")
cat("Doc IDs sample:", head(doc_ids, 3), "\n")

doc_meta <- read_csv(file.path(TMPDIR, "doc_meta_party.csv")) %>%
  filter(doc_party_session %in% doc_ids) %>%
  arrange(match(doc_party_session, doc_ids)) %>%
  mutate(
    bjp     = as.integer(party_family == "BJP"),
    inc     = as.integer(party_family == "INC"),
    left    = as.integer(party_family == "Left"),
    lok_num = as.integer(lok_no),
    session = as.integer(session_no)
  ) %>%
  as.data.frame()  # STM works better with data.frame than tibble

cat("Metadata rows:", nrow(doc_meta), "\n")

stm_input <- readCorpus(dtm, type = "slam")
cat("Documents:", length(stm_input$documents), "\n")
stopifnot(nrow(doc_meta) == length(stm_input$documents))

#}

# ============================================================
# SECTION 2: K selection (coherence vs. exclusivity)
# ============================================================
#{

# K=20 chosen based on LDA perplexity curve (A4) and interpretability
K_stm <- 20

#}

# ============================================================
# SECTION 3: Fit final STM
# ============================================================
#{

cat("Fitting STM K =", K_stm, "...\n")

stm_final <- stm(
  documents  = stm_input$documents,
  vocab      = stm_input$vocab,
  K          = K_stm,
  prevalence = ~ bjp + inc + left + lok_num + session,
  data       = doc_meta,
  init.type  = "Spectral",
  seed       = 42
)

saveRDS(stm_final, file.path(OUTDIR, "models", paste0("stm_k", K_stm, ".rds")))
cat("STM saved.\n")

#}

# ============================================================
# SECTION 4: Topic labels (FREX)
# ============================================================
#{

frex <- labelTopics(stm_final, n = 10)

topic_df <- purrr::map_dfr(1:K_stm, function(k) {
  tibble(topic = k,
         type  = rep(c("prob","frex","lift","score"), each = 10),
         term  = c(frex$prob[k,], frex$frex[k,], frex$lift[k,], frex$score[k,]))
})

write_csv(topic_df, file.path(OUTDIR, "tables", "stm_top_terms.csv"))

cat("\n=== STM FREX TERMS ===\n")
topic_df %>%
  filter(type == "frex") %>%
  group_by(topic) %>%
  summarise(frex = paste(term, collapse = ", ")) %>%
  print(n = Inf)

#}

# ============================================================
# SECTION 5: Covariate effects
# ============================================================
#{

effects <- estimateEffect(
  formula = 1:K_stm ~ bjp + inc + left + lok_num,
  stmobj  = stm_final,
  metadata = doc_meta
)
saveRDS(effects, file.path(TMPDIR, "stm_effects.rds"))

eff_df <- purrr::map_dfr(1:K_stm, function(k) {
  s <- summary(effects, topics = k)$tables[[1]]
  tibble(topic     = k,
         covariate = rownames(s),
         estimate  = s[,"Estimate"],
         se        = s[,"Std. Error"],
         pval      = s[,"Pr(>|t|)"]) %>%
    mutate(ci_lo = estimate - 1.96*se,
           ci_hi = estimate + 1.96*se,
           sig   = pval < 0.05)
})
write_csv(eff_df, file.path(OUTDIR, "tables", "stm_covariate_effects.csv"))

# Lok Sabha number effect (time trend)
p_lok <- eff_df %>%
  filter(covariate == "lok_num") %>%
  ggplot(aes(estimate, reorder(factor(topic), estimate),
             color = sig, xmin = ci_lo, xmax = ci_hi)) +
  geom_pointrange() +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_color_manual(values = c("FALSE"="grey70","TRUE"="#d73027"),
                     name = "p<0.05") +
  labs(title = "Topics Growing/Shrinking Across Lok Sabhas",
       subtitle = "Effect of Lok Sabha number on topic prevalence (16→17→18 LS)",
       x = "Marginal Effect", y = "Topic") +
  theme_minimal(base_size = 12)

ggsave(file.path(OUTDIR, "figures", "stm_effects_lok.png"),
       p_lok, width = 8, height = 6, dpi = 300)

# BJP vs INC contrast
p_contrast <- eff_df %>%
  filter(covariate %in% c("bjp","inc")) %>%
  select(topic, covariate, estimate) %>%
  pivot_wider(names_from = covariate, values_from = estimate) %>%
  ggplot(aes(inc, bjp, label = topic)) +
  geom_point(size = 3) +
  geom_text_repel(size = 3.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "BJP vs. INC Topic Contrast",
       x = "INC effect", y = "BJP effect") +
  theme_minimal(base_size = 12)

ggsave(file.path(OUTDIR, "figures", "stm_bjp_inc_contrast.png"),
       p_contrast, width = 7, height = 7, dpi = 300)

cat("\nA5 complete.\n")

#}
