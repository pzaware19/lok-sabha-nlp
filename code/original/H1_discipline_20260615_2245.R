# =============================================================================
# H1_discipline.R — Opposition Discipline Index
# Author: Piyush Zaware
# Updated: 2026-06-14
#
# PURPOSE:
#   Measures how coordinated each party's parliamentary questioning is.
#   A disciplined party's MPs ask about the same topics in similar language.
#   A diverse/undisciplined party has MPs pulling in different directions.
#
#   Method: For each MP with ≥5 starred questions, build a TF-IDF vector
#   over their question vocabulary. Compute each MP's cosine similarity to
#   their party's centroid (average) vector. The party discipline score is
#   the mean of these centroid similarities.
#
#   High discipline score → MPs converge on shared vocabulary → coordinated
#   Low discipline score  → MPs diverge → independent or geographically driven
#
# OUTPUTS:
#   output/figures/discipline_party.png
#   output/figures/discipline_heatmap.png
#   output/figures/discipline_mp_scatter.png
#   output/tables/discipline_scores.csv
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

pkgs <- c("arrow","tidyverse","tidytext","patchwork","ggrepel","scales","stopwords")
to_inst <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_inst)) install.packages(to_inst)
suppressPackageStartupMessages(lapply(pkgs, library, character.only = TRUE))

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")
NAVY   <- "#0D1B2A"; SAFFRON <- "#FF6B35"; GREEN <- "#138808"
BJP_COL <- "#FF9933"; INC_COL <- "#19AAED"

source(file.path(CODDIR, "._stop_words.R"))

# ============================================================
# SECTION 1: Load and prep
# ============================================================
#{
cat("Loading data...\n")
parquet_files <- list.files(TMPDIR, pattern = "train-.*\\.parquet$", full.names = TRUE)
raw <- purrr::map_dfr(parquet_files, function(f)
  read_parquet(f, col_select = c("id","lok_no","type","members","question_text")))
starred <- raw %>% filter(type == "STARRED", lok_no >= 16)

strip_hon <- function(s) {
  s <- str_to_upper(str_squish(s))
  s <- str_remove_all(s, "\\b(SHRIMATI|SMT\\.?|KUMARI|MRS\\.?|MS\\.?|DR\\.?|PROF\\.?|SH\\.?|SHRI\\.?)\\b")
  str_squish(s)
}
norm_fl <- function(s) {
  parts <- str_split(str_squish(s), "\\s+")[[1]]
  if (length(parts) <= 2) return(s)
  paste(parts[1], parts[length(parts)])
}

lookup <- read_csv(file.path(INPDIR, "mp_party_lookup.csv"), show_col_types = FALSE) %>%
  mutate(
    mp_key  = str_to_upper(str_squish(mp_name)),
    mp_norm = vapply(vapply(mp_key, strip_hon, character(1)), norm_fl, character(1))
  ) %>%
  arrange(desc(lok_no)) %>%
  distinct(mp_norm, .keep_all = TRUE)

mp_party <- setNames(lookup$party_family, lookup$mp_norm)

starred <- starred %>%
  mutate(
    primary_raw  = map_chr(members, function(x)
      tryCatch(str_squish(as.character(list(x)[[1]])[1]), error = function(e) NA_character_)),
    primary_norm = vapply(vapply(primary_raw, strip_hon, character(1)), norm_fl, character(1)),
    party_family = mp_party[primary_norm]
  ) %>%
  filter(!is.na(party_family), !is.na(question_text), !is.na(primary_norm))

cat(sprintf("  Starred (party-matched): %d\n", nrow(starred)))
#}

# ============================================================
# SECTION 2: Build MP-level TF-IDF vectors
# ============================================================
#{
cat("Building MP word counts...\n")
MIN_Q <- 5   # minimum questions for an MP to be included

# Count questions per MP × Lok Sabha
mp_q_counts <- starred %>%
  count(primary_norm, party_family, lok_no, name = "n_questions")

eligible_mps <- mp_q_counts %>%
  filter(n_questions >= MIN_Q)

cat(sprintf("  MPs with >= %d questions: %d\n", MIN_Q, nrow(eligible_mps)))

# Word counts per (MP, lok_no)
mp_words <- starred %>%
  semi_join(eligible_mps, by = c("primary_norm","lok_no")) %>%
  unnest_tokens(word, question_text) %>%
  filter(!word %in% COMBINED_STOP,
         str_detect(word, "^[a-z]+$"), nchar(word) >= 5) %>%
  count(primary_norm, party_family, lok_no, word, name = "n")

# TF-IDF: document = individual MP × Lok Sabha
mp_tfidf <- mp_words %>%
  mutate(mp_ls = paste(primary_norm, lok_no, sep = "_")) %>%
  bind_tf_idf(word, mp_ls, n)

cat(sprintf("  MP-level vocab: %d unique words\n", n_distinct(mp_tfidf$word)))
#}

# ============================================================
# SECTION 3: Compute discipline scores
# ============================================================
#{
cat("Computing discipline scores...\n")

cosine_to_centroid <- function(df) {
  # df has columns: primary_norm, word, tf_idf
  # Returns per-MP cosine similarity to party centroid

  # Build wide matrix: rows = MPs, cols = words
  vocab  <- unique(df$word)
  mp_ids <- unique(df$primary_norm)

  # Centroid = mean TF-IDF per word across all MPs
  centroid <- df %>%
    group_by(word) %>%
    summarise(c_val = mean(tf_idf), .groups = "drop")

  # Per-MP similarity to centroid
  purrr::map_dfr(mp_ids, function(mp) {
    mp_vec  <- df %>% filter(primary_norm == mp) %>%
      select(word, tf_idf) %>% deframe()
    cen_vec <- centroid %>% deframe()
    shared  <- intersect(names(mp_vec), names(cen_vec))
    if (length(shared) < 3) return(tibble(primary_norm = mp, cosine = NA_real_))
    num   <- sum(mp_vec[shared] * cen_vec[shared])
    denom <- sqrt(sum(mp_vec^2)) * sqrt(sum(cen_vec^2))
    tibble(primary_norm = mp, cosine = if (denom > 0) num / denom else 0)
  })
}

# Run for each party × Lok Sabha with >= 3 MPs
party_ls_groups <- eligible_mps %>%
  count(party_family, lok_no) %>%
  filter(n >= 3)

discipline_mp <- purrr::pmap_dfr(party_ls_groups, function(party_family, lok_no, n) {
  cat(sprintf("  %s LS%d (%d MPs)...\n", party_family, lok_no, n))
  sub <- mp_tfidf %>%
    filter(party_family == .env$party_family, lok_no == .env$lok_no)
  if (nrow(sub) == 0) return(NULL)
  cosine_to_centroid(sub) %>%
    mutate(party_family = .env$party_family, lok_no = .env$lok_no,
           n_mps = n)
})

discipline_party <- discipline_mp %>%
  filter(!is.na(cosine)) %>%
  group_by(party_family, lok_no, n_mps) %>%
  summarise(
    discipline    = mean(cosine),
    discipline_sd = sd(cosine),
    n_mps_used    = n(),
    .groups       = "drop"
  )

write_csv(discipline_mp %>% filter(!is.na(cosine)),
          file.path(TABDIR, "discipline_mp.csv"))
write_csv(discipline_party, file.path(TABDIR, "discipline_scores.csv"))

cat("\n=== Discipline scores by party (averaged across Lok Sabhas) ===\n")
discipline_party %>%
  group_by(party_family) %>%
  summarise(mean_disc = mean(discipline), n_ls = n(), .groups = "drop") %>%
  arrange(desc(mean_disc)) %>%
  print(n = 20)
#}

# ============================================================
# SECTION 4: Figure 1 — Party discipline bar chart
# ============================================================
#{
cat("\nPlotting party discipline chart...\n")

# Average across Lok Sabhas, require data in >= 2 LS
party_avg <- discipline_party %>%
  group_by(party_family) %>%
  filter(sum(n_mps_used) >= 10) %>%
  summarise(
    discipline    = weighted.mean(discipline, n_mps_used),
    n_mps         = sum(n_mps_used),
    .groups       = "drop"
  ) %>%
  mutate(
    party_type = case_when(
      party_family == "BJP"  ~ "Ruling (BJP)",
      party_family == "INC"  ~ "Main Opposition (INC)",
      TRUE                   ~ "Regional / Other"
    ),
    party_family = fct_reorder(party_family, discipline)
  )

p_disc_bar <- ggplot(party_avg,
                     aes(x = discipline, y = party_family, fill = party_type)) +
  geom_col(width = 0.75) +
  geom_text(aes(label = sprintf("%.3f  (%d MPs)", discipline, n_mps)),
            hjust = -0.05, size = 3.2, colour = "grey30") +
  scale_fill_manual(
    values = c("Ruling (BJP)" = BJP_COL,
               "Main Opposition (INC)" = INC_COL,
               "Regional / Other" = "grey60"),
    name = NULL
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25)),
                     labels = scales::number_format(accuracy = 0.001)) +
  labs(
    title    = "Which party's MPs ask the most similar questions?",
    subtitle = "Average cosine similarity of each MP's questions to their party's centroid vector.\nHigher = MPs converge on shared vocabulary = more coordinated questioning.",
    x = "Discipline score (mean cosine similarity to party centroid)", y = NULL,
    caption  = "MPs with fewer than 5 starred questions excluded. 16th, 17th and 18th Lok Sabha combined."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", colour = NAVY, size = 13),
    plot.subtitle   = element_text(colour = "grey40", size = 9),
    legend.position = "bottom"
  )

ggsave(file.path(FIGDIR, "discipline_party.png"),
       p_disc_bar, width = 11, height = 7, dpi = 180)
cat("Saved: discipline_party.png\n")
#}

# ============================================================
# SECTION 5: Figure 2 — Discipline heatmap (party × Lok Sabha)
# ============================================================
#{
cat("Plotting discipline heatmap...\n")

heat_data <- discipline_party %>%
  filter(n_mps_used >= 5) %>%
  mutate(
    ls_label = paste0(lok_no, "th LS\n(",
                      case_when(lok_no == 16 ~ "2014-19",
                                lok_no == 17 ~ "2019-24",
                                lok_no == 18 ~ "2024-"), ")"),
    party_family = fct_reorder(party_family, discipline, .fun = mean, .desc = TRUE)
  )

p_disc_heat <- ggplot(heat_data,
                      aes(x = ls_label, y = party_family, fill = discipline)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.3f\n(%d MPs)", discipline, n_mps_used)),
            size = 2.8, colour = "white", fontface = "bold") +
  scale_fill_gradient2(
    low = "#FFF5E6", mid = "#FF9933", high = "#7B1FA2",
    midpoint = 0.25,
    name = "Discipline\nscore",
    limits = c(0.1, 0.5), oob = scales::squish
  ) +
  labs(
    title    = "Party discipline across three Lok Sabhas",
    subtitle = "Cosine similarity of MP questions to party centroid. Higher = more coordinated.",
    x = NULL, y = NULL,
    caption  = "Parties with fewer than 5 eligible MPs in a Lok Sabha excluded."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", colour = NAVY, size = 13),
    plot.subtitle   = element_text(colour = "grey40", size = 9),
    panel.grid      = element_blank(),
    legend.position = "right"
  )

ggsave(file.path(FIGDIR, "discipline_heatmap.png"),
       p_disc_heat, width = 10, height = 6, dpi = 180)
cat("Saved: discipline_heatmap.png\n")
#}

# ============================================================
# SECTION 6: Figure 3 — Within-party MP spread (BJP vs INC)
# ============================================================
#{
cat("Plotting within-party MP spread...\n")

mp_spread <- discipline_mp %>%
  filter(!is.na(cosine), party_family %in% c("BJP","INC")) %>%
  left_join(eligible_mps, by = c("primary_norm","party_family","lok_no")) %>%
  mutate(
    ls_label = paste0(lok_no, "th LS"),
    label    = if_else(cosine < quantile(cosine, 0.05, na.rm = TRUE) |
                       cosine > quantile(cosine, 0.95, na.rm = TRUE),
                       str_to_title(primary_norm), NA_character_)
  )

p_spread <- ggplot(mp_spread,
                   aes(x = n_questions, y = cosine,
                       colour = party_family, size = n_questions)) +
  geom_point(alpha = 0.6) +
  ggrepel::geom_text_repel(aes(label = label), size = 2.5,
                            max.overlaps = 12, segment.size = 0.3,
                            show.legend = FALSE) +
  facet_wrap(~ls_label, ncol = 3) +
  scale_colour_manual(values = c("BJP" = BJP_COL, "INC" = INC_COL), name = NULL) +
  scale_size_continuous(range = c(1, 5), guide = "none") +
  scale_x_log10() +
  labs(
    title    = "Which BJP and INC MPs diverge most from their party line?",
    subtitle = "Each dot is one MP. Y-axis = cosine similarity to party centroid.\nLow y = questions are distinctly different from the party's average vocabulary.",
    x = "Number of starred questions (log scale)", y = "Similarity to party centroid",
    caption  = "Labels = top and bottom 5% outliers by cosine similarity."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", colour = NAVY, size = 12),
    plot.subtitle   = element_text(colour = "grey40", size = 9),
    legend.position = "bottom",
    strip.text      = element_text(face = "bold")
  )

ggsave(file.path(FIGDIR, "discipline_mp_scatter.png"),
       p_spread, width = 12, height = 6, dpi = 180)
cat("Saved: discipline_mp_scatter.png\n")
#}

# ============================================================
# SECTION 7: Figure 4 — BJP discipline: ruling vs earlier (change over time)
# ============================================================
#{
cat("Plotting BJP vs INC discipline over time...\n")

time_data <- discipline_party %>%
  filter(party_family %in% c("BJP","INC"), n_mps_used >= 5) %>%
  mutate(
    ls_label = factor(paste0(lok_no, "th LS"),
                      levels = c("16th LS","17th LS","18th LS"))
  )

p_time <- ggplot(time_data,
                 aes(x = ls_label, y = discipline,
                     colour = party_family, group = party_family)) +
  geom_line(linewidth = 1.2) +
  geom_point(aes(size = n_mps_used), alpha = 0.9) +
  geom_errorbar(aes(ymin = discipline - discipline_sd/sqrt(n_mps_used),
                    ymax = discipline + discipline_sd/sqrt(n_mps_used)),
                width = 0.1, linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.3f", discipline)),
            vjust = -1.2, size = 3.5, fontface = "bold") +
  scale_colour_manual(values = c("BJP" = BJP_COL, "INC" = INC_COL), name = NULL) +
  scale_size_continuous(range = c(4, 8), name = "MPs included") +
  scale_y_continuous(limits = c(0.1, 0.6)) +
  labs(
    title    = "BJP vs INC: how discipline evolves across Lok Sabhas",
    subtitle = "Does the ruling party grow more or less coordinated over time?\nError bars show standard error of mean.",
    x = NULL, y = "Discipline score (cosine similarity to centroid)",
    caption  = "16th LS = BJP's first term; 17th = second term; 18th = third term (ongoing)."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", colour = NAVY, size = 12),
    plot.subtitle   = element_text(colour = "grey40", size = 9),
    legend.position = "bottom"
  )

ggsave(file.path(FIGDIR, "discipline_time.png"),
       p_time, width = 9, height = 6, dpi = 180)
cat("Saved: discipline_time.png\n")
#}

cat("\n=== H1 complete ===\n")
