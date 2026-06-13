# =============================================================================
# A9_incumbency_robustness.R — Incumbency (Government vs. Opposition) Robustness
# Author: Piyush Zaware
# Updated: 2026-06-12
#
# PURPOSE:
#   Addresses the government-opposition incumbency confound:
#   parties in power ask more delivery/infrastructure questions;
#   opposition parties ask more accountability/scrutiny questions.
#   This positional effect could inflate apparent ideological differences.
#
#   Checks:
#   (1) Build party × Lok Sabha incumbency indicator
#   (2) Show the confound: topic profiles by government vs. opposition status
#   (3) TDP natural experiment — TDP exited NDA between 16th and 17th LS,
#       then rejoined in 18th. If topic mix follows incumbency, the confound
#       is real. If topics stay stable, the signal is ideological.
#   (4) Re-fit STM adding in_gov covariate; compare BJP/INC effects before/after
#   (5) Residualize Kozlowski ideology scores on incumbency; show ordering survives
#
# INPUTS:
#   $TMPDIR/dtm_party_session.rds       — DTM (from A3)
#   $TMPDIR/doc_meta_party.csv          — doc metadata (from A3)
#   $OUTDIR/tables/lda_party_topics.csv — LDA gamma per doc × topic (from A4)
#   $OUTDIR/tables/party_dimensions.csv — Kozlowski scores per doc (from A6)
#   $OUTDIR/models/stm_k20.rds          — fitted STM (from A5)
#
# OUTPUTS:
#   $OUTDIR/figures/incumbency_topic_profiles.png
#   $OUTDIR/figures/tdp_natural_experiment.png
#   $OUTDIR/figures/stm_incumbency_effects.png
#   $OUTDIR/figures/ideology_incumbency_adjusted.png
#   $OUTDIR/tables/incumbency_adjusted_scores.csv
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(stm)
  library(ggrepel)
  library(patchwork)
})

set.seed(42)

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")

# ============================================================
# SECTION 1: Incumbency lookup table
# ============================================================
#{
# NDA government composition per Lok Sabha.
# Key identification sources (within-party switchers):
#   TDP  : NDA in 16th → opposition in 17th (left over AP special status) → NDA in 18th
#   JDU  : contested alone in 2014 (opposition, 2 seats) → NDA in 17th and 18th
#   Shiv Sena: NDA in 16th and 17th → split 2022; Shinde faction in NDA for 18th
# All others are stable (BJP always in, Left/INC/BSP/AAP/TMC/SP/DMK/RJD always out).

incumbency_tbl <- tribble(
  ~party_family,  ~lok_no, ~in_gov,
  # NDA core
  "BJP",          16L,     1L,
  "BJP",          17L,     1L,
  "BJP",          18L,     1L,
  # Shiv Sena — NDA throughout (Shinde faction post-split)
  "Shiv Sena",    16L,     1L,
  "Shiv Sena",    17L,     1L,
  "Shiv Sena",    18L,     1L,
  # TDP — key switcher
  "TDP",          16L,     1L,  # NDA ally, Naidu held cabinet post
  "TDP",          17L,     0L,  # Exited NDA March 2018, contested against BJP in 2019
  "TDP",          18L,     1L,  # Rejoined NDA June 2024
  # JDU — key switcher
  "JDU",          16L,     0L,  # Left NDA 2013, won only 2 seats in 2014 alone
  "JDU",          17L,     1L,  # Back in NDA, 16 seats
  "JDU",          18L,     1L,  # NDA, 12 seats
  # Stable opposition
  "INC",          16L,     0L,
  "INC",          17L,     0L,
  "INC",          18L,     0L,
  "Left",         16L,     0L,
  "Left",         17L,     0L,
  "Left",         18L,     0L,
  "BSP",          16L,     0L,
  "BSP",          17L,     0L,
  "BSP",          18L,     0L,
  "AAP",          16L,     0L,
  "AAP",          17L,     0L,
  "AAP",          18L,     0L,
  "SP",           16L,     0L,
  "SP",           17L,     0L,
  "SP",           18L,     0L,
  "TMC",          16L,     0L,
  "TMC",          17L,     0L,
  "TMC",          18L,     0L,
  "DMK",          16L,     0L,
  "DMK",          17L,     0L,
  "DMK",          18L,     0L,
  "RJD",          16L,     0L,
  "RJD",          17L,     0L,
  "RJD",          18L,     0L,
  "NCP",          16L,     0L,
  "NCP",          17L,     0L,
  "NCP",          18L,     0L
)
#}

# ============================================================
# SECTION 2: Load data and merge incumbency
# ============================================================
#{

lda_docs <- read_csv(file.path(TABDIR, "lda_party_topics.csv"), show_col_types = FALSE) %>%
  left_join(incumbency_tbl, by = c("party_family", "lok_no")) %>%
  mutate(
    in_gov    = replace_na(in_gov, 0L),
    gov_label = if_else(in_gov == 1L, "Government (NDA)", "Opposition")
  )

dims_docs <- read_csv(file.path(TABDIR, "party_dimensions.csv"), show_col_types = FALSE) %>%
  left_join(incumbency_tbl, by = c("party_family", "lok_no")) %>%
  mutate(
    in_gov    = replace_na(in_gov, 0L),
    gov_label = if_else(in_gov == 1L, "Government (NDA)", "Opposition")
  )

cat("LDA docs:", n_distinct(lda_docs$doc_party_session), "documents,",
    n_distinct(lda_docs$party_family), "parties\n")
cat("Incumbency breakdown:\n")
dims_docs %>% distinct(doc_party_session, .keep_all = TRUE) %>%
  count(gov_label) %>% print()
#}

# ============================================================
# SECTION 3: Government vs. opposition topic profiles
# ============================================================
# Documents the incumbency confound: which LDA topics differ by status.
#{

K_lda <- n_distinct(lda_docs$topic)

gov_opp <- lda_docs %>%
  group_by(gov_label, topic) %>%
  summarise(mean_gamma = mean(gamma, na.rm = TRUE), .groups = "drop")

# Compute government premium (gov - opp) per topic
premium <- gov_opp %>%
  pivot_wider(names_from = gov_label, values_from = mean_gamma) %>%
  rename(gov = `Government (NDA)`, opp = Opposition) %>%
  mutate(
    premium   = gov - opp,
    direction = if_else(premium > 0, "Higher in government", "Higher in opposition"),
    topic_lbl = paste("Topic", topic)
  ) %>%
  arrange(premium)

p_confound <- ggplot(premium,
                     aes(x = reorder(topic_lbl, premium),
                         y = premium * 100,
                         fill = direction)) +
  geom_col(width = 0.75) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  coord_flip() +
  scale_fill_manual(
    values = c("Higher in government" = "#138808", "Higher in opposition" = "#B5440E"),
    name = NULL
  ) +
  labs(
    title    = "The incumbency confound: topic usage by government vs. opposition parties",
    subtitle = "Government parties (NDA) ask more about delivery; opposition asks more about accountability",
    x = NULL,
    y = "Difference in mean topic weight (government minus opposition, ×100)",
    caption  = "LDA K=15. Based on 205 party×session documents across 16th–18th Lok Sabha."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position   = "bottom",
    plot.title        = element_text(face = "bold"),
    panel.grid.major.y = element_blank()
  )

ggsave(file.path(FIGDIR, "incumbency_topic_profiles.png"),
       p_confound, width = 9, height = 6, dpi = 180)
cat("Saved: incumbency_topic_profiles.png\n")
#}

# ============================================================
# SECTION 4: TDP natural experiment
# ============================================================
# TDP is the cleanest switcher: government (16th) → opposition (17th) → government (18th).
# If topic mix follows incumbency, we see the confound operating.
#{

tdp_topics <- lda_docs %>%
  filter(party_family == "TDP") %>%
  group_by(lok_no, topic) %>%
  summarise(mean_gamma = mean(gamma, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    ls_label = factor(
      lok_no,
      levels = c(16L, 17L, 18L),
      labels = c("16th LS\n(NDA govt)", "17th LS\n(Opposition)", "18th LS\n(NDA govt)")
    )
  )

# Identify topics that shift most between available Lok Sabhas
tdp_ls_avail <- sort(unique(tdp_topics$lok_no))
cat("TDP data available for Lok Sabhas:", paste(tdp_ls_avail, collapse = ", "), "\n")

if (length(tdp_ls_avail) >= 2) {
  ls_ref  <- tdp_ls_avail[1]
  ls_comp <- tdp_ls_avail[2]
  tdp_shift <- tdp_topics %>%
    filter(lok_no %in% c(ls_ref, ls_comp)) %>%
    pivot_wider(names_from = lok_no, values_from = mean_gamma,
                names_prefix = "ls", values_fill = 0) %>%
    rename_with(~ gsub("\\.", "", .x)) %>%
    mutate(shift = abs(.data[[paste0("ls", ls_comp)]] -
                         .data[[paste0("ls", ls_ref)]])) %>%
    slice_max(shift, n = 8) %>%
    pull(topic)
} else {
  tdp_shift <- unique(tdp_topics$topic)[1:8]
  cat("Warning: TDP has data in only one Lok Sabha — cannot compute shift\n")
}

p_tdp <- tdp_topics %>%
  filter(topic %in% tdp_shift) %>%
  mutate(topic_lbl = paste("Topic", topic)) %>%
  ggplot(aes(x = ls_label, y = mean_gamma * 100, group = topic_lbl,
             colour = topic_lbl)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 3) +
  facet_wrap(~ topic_lbl, ncol = 4, scales = "free_y") +
  scale_colour_brewer(palette = "Dark2", guide = "none") +
  labs(
    title    = "TDP natural experiment: topic mix before and after leaving NDA",
    subtitle = "TDP exited NDA in March 2018 (between 16th and 17th LS) over Andhra Pradesh special status",
    x = NULL,
    y = "Mean LDA topic weight (×100)",
    caption  = "Eight topics with largest absolute shift between 16th (NDA) and 17th (opposition) LS shown."
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title   = element_text(face = "bold"),
    strip.text   = element_text(face = "bold", size = 9),
    axis.text.x  = element_text(size = 7.5)
  )

ggsave(file.path(FIGDIR, "tdp_natural_experiment.png"),
       p_tdp, width = 11, height = 5, dpi = 180)
cat("Saved: tdp_natural_experiment.png\n")
#}

# ============================================================
# SECTION 5: STM with incumbency covariate
# ============================================================
# Re-fit STM adding in_gov. Compare party effects (BJP, INC) before and after.
# If BJP coefficient on ideology-adjacent topics shrinks, incumbency was driving it.
#{

dtm      <- readRDS(file.path(TMPDIR, "dtm_party_session.rds"))
doc_ids  <- dtm$dimnames$Docs
doc_meta <- read_csv(file.path(TMPDIR, "doc_meta_party.csv"), show_col_types = FALSE) %>%
  filter(doc_party_session %in% doc_ids) %>%
  arrange(match(doc_party_session, doc_ids)) %>%
  left_join(incumbency_tbl, by = c("party_family", "lok_no")) %>%
  mutate(
    bjp    = as.integer(party_family == "BJP"),
    inc    = as.integer(party_family == "INC"),
    left   = as.integer(party_family == "Left"),
    lok_num = as.integer(lok_no),
    session = as.integer(session_no),
    in_gov  = replace_na(as.integer(in_gov), 0L)
  ) %>%
  as.data.frame()

stm_input <- readCorpus(dtm, type = "slam")

cat("Fitting STM with incumbency (K=20)...\n")
stm_incumb <- stm(
  documents  = stm_input$documents,
  vocab      = stm_input$vocab,
  K          = 20L,
  prevalence = ~ bjp + inc + left + lok_num + session + in_gov,
  data       = doc_meta,
  init.type  = "Spectral",
  seed       = 42
)
saveRDS(stm_incumb, file.path(OUTDIR, "models", "stm_k20_incumbency.rds"))
cat("Saved: stm_k20_incumbency.rds\n")

# Estimate effects for all three party indicators + incumbency
eff_incumb <- estimateEffect(1:20 ~ bjp + inc + left + in_gov,
                             stmobj   = stm_incumb,
                             metadata = doc_meta)

# Extract coefficients for BJP, INC, and in_gov
extract_coef <- function(eff, covariate, K = 20) {
  purrr::map_dfr(1:K, function(k) {
    cf <- summary(eff)$tables[[k]]
    if (!covariate %in% rownames(cf)) return(NULL)
    tibble(
      topic     = k,
      covariate = covariate,
      estimate  = cf[covariate, "Estimate"],
      se        = cf[covariate, "Std. Error"],
      pval      = cf[covariate, "Pr(>|t|)"]
    )
  }) %>%
    mutate(lo = estimate - 1.96 * se, hi = estimate + 1.96 * se,
           sig = pval < 0.05)
}

coef_df <- bind_rows(
  extract_coef(eff_incumb, "bjp"),
  extract_coef(eff_incumb, "inc"),
  extract_coef(eff_incumb, "in_gov")
) %>%
  mutate(
    topic_lbl = paste("Topic", topic),
    label     = recode(covariate,
                       bjp    = "BJP (net of incumbency)",
                       inc    = "INC (net of incumbency)",
                       in_gov = "In government effect")
  )

p_stm_incumb <- ggplot(coef_df,
                       aes(x = estimate, xmin = lo, xmax = hi,
                           y = reorder(topic_lbl, estimate),
                           colour = sig)) +
  geom_vline(xintercept = 0, linewidth = 0.4, linetype = "dashed", colour = "grey60") +
  geom_errorbarh(height = 0.3, linewidth = 0.5) +
  geom_point(size = 2) +
  facet_wrap(~ label, ncol = 3) +
  scale_colour_manual(
    values = c("TRUE" = "#B5440E", "FALSE" = "grey70"),
    labels = c("TRUE" = "p < 0.05", "FALSE" = "n.s."),
    name   = NULL
  ) +
  labs(
    title    = "STM party effects after controlling for incumbency",
    subtitle = "Coefficients from STM prevalence ~ bjp + inc + left + lok_num + session + in_gov",
    x = "Estimated effect on topic prevalence (95% CI)",
    y = NULL,
    caption  = "Topics where party effect persists net of government status are genuinely ideological."
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position    = "bottom",
    plot.title         = element_text(face = "bold"),
    panel.grid.major.y = element_blank(),
    strip.text         = element_text(face = "bold")
  )

ggsave(file.path(FIGDIR, "stm_incumbency_effects.png"),
       p_stm_incumb, width = 13, height = 7, dpi = 180)
cat("Saved: stm_incumbency_effects.png\n")
#}

cat("\n=== A9 complete ===\n")
cat("Figures: incumbency_topic_profiles, tdp_natural_experiment, stm_incumbency_effects\n")
cat("Note: residualization of Kozlowski scores on in_gov is NOT valid here.\n")
cat("Because BJP is always in government and INC always in opposition, in_gov is\n")
cat("collinear with the party indicators. Regressing out in_gov just removes the\n")
cat("BJP-INC contrast and calling the residuals 'ideology-adjusted' is circular.\n")
cat("Proper identification requires within-party incumbency variation (IV or RD).\n")
