# =============================================================================
# I1_electoral_incentives.R
# Author: Piyush Zaware
# Updated: 2026-06-14
#
# Goal: Link election margin data (2014, 2019) to MP question behavior.
#       Tests whether marginal MPs question more, and differently, than MPs
#       from safe seats.
#
# Inputs:
#   $INPDIR/mp_party_lookup.csv
#   $INPDIR/election_2014_bhavnani.tab
#   $INPDIR/election_2019.csv
#   $TMPDIR/train-*.parquet
#
# Outputs:
#   $OUTDIR/figures/elec_margin_volume.png
#   $OUTDIR/figures/elec_margin_vocab.png
#   $OUTDIR/figures/elec_temporal_spike.png
#   $OUTDIR/figures/elec_safe_vs_marginal.png
#   $OUTDIR/tables/electoral_scores.csv
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

pkgs <- c("arrow", "tidyverse", "tidytext", "ggrepel", "scales")
to_inst <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_inst)) install.packages(to_inst)
suppressPackageStartupMessages(lapply(pkgs, library, character.only = TRUE))

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")

source(file.path(CODDIR, "._stop_words.R"))

MARGINAL_THRESH <- 0.10
SAFE_THRESH     <- 0.25
MIN_Q           <- 3

# =============================================================================
# SECTION 1: Load election data and compute margins
# =============================================================================
#{
cat("\n[I1] Computing election margins...\n")

# --- 2014 (Bhavnani dataset) ---
e14 <- read_tsv(file.path(INPDIR, "election_2014_bhavnani.tab"),
                col_types = cols(.default = "c"), show_col_types = FALSE) %>%
  filter(year == "2014") %>%
  mutate(votes = as.integer(totvotpoll), electors = as.integer(electors)) %>%
  filter(!is.na(votes), electors > 0)

e14_margins <- e14 %>%
  group_by(pc_name) %>%
  arrange(desc(votes)) %>%
  summarise(
    winner_votes   = first(votes),
    runnerup_votes = nth(votes, 2),
    total_electors = first(electors),
    .groups = "drop"
  ) %>%
  filter(!is.na(runnerup_votes)) %>%
  mutate(
    margin_pct         = (winner_votes - runnerup_votes) / total_electors,
    vote_share         = winner_votes / total_electors,
    constituency_norm  = pc_name %>%
      str_to_upper() %>%
      str_remove_all("\\s*\\(SC\\)|\\s*\\(ST\\)") %>%
      str_replace_all("–|—", "-") %>%
      str_trim(),
    election_year = 2014L
  ) %>%
  select(constituency_norm, margin_pct, vote_share, election_year)

# --- 2019 (pratapvardhan/ECI) ---
e19 <- read_csv(file.path(INPDIR, "election_2019.csv"),
                col_types = cols(.default = "c"), show_col_types = FALSE) %>%
  mutate(votes = as.integer(`Total Votes`)) %>%
  filter(!is.na(votes))

e19_margins <- e19 %>%
  group_by(Constituency) %>%
  arrange(desc(votes)) %>%
  summarise(
    winner_votes   = first(votes),
    runnerup_votes = nth(votes, 2),
    total_votes    = sum(votes, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(!is.na(runnerup_votes), total_votes > 0) %>%
  mutate(
    margin_pct        = (winner_votes - runnerup_votes) / total_votes,
    vote_share        = winner_votes / total_votes,
    constituency_norm = Constituency %>%
      str_to_upper() %>%
      str_remove_all("\\s*\\(SC\\)|\\s*\\(ST\\)") %>%
      str_replace_all("–|—", "-") %>%
      str_trim(),
    election_year = 2019L
  ) %>%
  select(constituency_norm, margin_pct, vote_share, election_year)

margins_all <- bind_rows(e14_margins, e19_margins)
cat("  2014:", nrow(e14_margins), "constituencies\n")
cat("  2019:", nrow(e19_margins), "constituencies\n")
#}

# =============================================================================
# SECTION 2: Load and normalize MP lookup + merge margins
# =============================================================================
#{
cat("[I1] Merging margins with MP lookup...\n")

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

lookup <- read_csv(file.path(INPDIR, "mp_party_lookup.csv"),
                   show_col_types = FALSE) %>%
  mutate(
    mp_norm = vapply(vapply(str_to_upper(str_squish(mp_name)),
                            strip_hon, character(1)),
                     norm_fl, character(1)),
    constituency_norm = constituency %>%
      str_to_upper() %>%
      str_remove_all("\\s*\\(SC\\)|\\s*\\(ST\\)") %>%
      str_replace_all("–|—", "-") %>%
      str_trim(),
    election_year = case_when(
      lok_no == 16L ~ 2014L,
      lok_no == 17L ~ 2019L,
      TRUE          ~ NA_integer_
    )
  ) %>%
  filter(!is.na(election_year))

mp_with_margin <- lookup %>%
  left_join(margins_all, by = c("constituency_norm", "election_year")) %>%
  filter(!is.na(margin_pct)) %>%
  mutate(
    seat_type = case_when(
      margin_pct < MARGINAL_THRESH ~ "Marginal",
      margin_pct > SAFE_THRESH     ~ "Safe",
      TRUE                          ~ "Competitive"
    )
  ) %>%
  select(mp_norm, lok_no, party_family, margin_pct, vote_share, seat_type)

cat("  MPs with margin data:", nrow(mp_with_margin), "\n")
cat("  Seat type distribution:\n")
print(table(mp_with_margin$seat_type))
#}

# =============================================================================
# SECTION 3: Load questions and join
# =============================================================================
#{
cat("[I1] Loading questions from parquet...\n")

parquet_files <- list.files(TMPDIR, pattern = "train-.*\\.parquet$", full.names = TRUE)
raw <- map_dfr(parquet_files, function(f)
  read_parquet(f, col_select = c("lok_no", "session_no", "type",
                                  "members", "question_text", "ministry")))

starred <- raw %>%
  filter(type == "STARRED", lok_no %in% c(16L, 17L)) %>%
  mutate(
    primary_raw  = map_chr(members, function(x)
      tryCatch(str_squish(as.character(list(x)[[1]])[1]), error = function(e) NA_character_)),
    primary_norm = vapply(vapply(replace_na(primary_raw, ""),
                                  strip_hon, character(1)),
                           norm_fl, character(1))
  ) %>%
  filter(primary_norm != "")

# Per-MP question count
mp_q_counts <- starred %>%
  count(primary_norm, lok_no, name = "n_questions")

# Merge with margin data
mp_data <- mp_with_margin %>%
  left_join(mp_q_counts, by = c("mp_norm" = "primary_norm", "lok_no")) %>%
  mutate(n_questions = replace_na(n_questions, 0)) %>%
  filter(n_questions >= MIN_Q)

cat("  MPs with margin + questions:", nrow(mp_data), "\n")

# Join questions with seat type for vocabulary analysis
starred_with_margin <- starred %>%
  inner_join(
    mp_data %>% select(mp_norm, lok_no, seat_type),
    by = c("primary_norm" = "mp_norm", "lok_no")
  )
#}

# =============================================================================
# SECTION 4: Figure 1 — Margin vs question volume
# =============================================================================
#{
cat("[I1] Figure 1: margin vs question volume...\n")

decile_avg <- mp_data %>%
  mutate(margin_decile = ntile(margin_pct, 10)) %>%
  group_by(margin_decile) %>%
  summarise(
    avg_q    = mean(n_questions),
    se_q     = sd(n_questions) / sqrt(n()),
    avg_marg = mean(margin_pct),
    n        = n(),
    .groups  = "drop"
  )

# OLS slope for annotation
lm_fit <- lm(n_questions ~ margin_pct, data = mp_data)
slope_str <- sprintf("OLS slope: %.1f fewer questions per 10pp margin",
                     -coef(lm_fit)["margin_pct"] * 0.10)

p1 <- ggplot(decile_avg, aes(x = avg_marg * 100, y = avg_q)) +
  geom_ribbon(aes(ymin = avg_q - 1.96 * se_q, ymax = avg_q + 1.96 * se_q),
              fill = "#2c7bb6", alpha = 0.15) +
  geom_line(color = "#2c7bb6", linewidth = 1) +
  geom_point(aes(size = n), color = "#2c7bb6", alpha = 0.85) +
  geom_smooth(data = mp_data, aes(x = margin_pct * 100, y = n_questions),
              method = "lm", se = FALSE, color = "#d7191c",
              linetype = "dashed", linewidth = 0.8, inherit.aes = FALSE) +
  annotate("text", x = 30, y = max(decile_avg$avg_q) * 0.95,
           label = slope_str, hjust = 1, size = 3.5, color = "#d7191c") +
  scale_size_continuous(range = c(2, 6), guide = "none") +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Marginal seat MPs ask fewer starred questions",
    subtitle = "Average starred questions per MP by margin of victory decile (16th and 17th Lok Sabha)",
    x        = "Margin of victory (% of total votes)",
    y        = "Average starred questions per MP",
    caption  = "Each point = one margin decile; n = MPs in decile. Dashed line = OLS fit.\nSource: Lok Sabha questions database merged with ECI 2014 and 2019 election results."
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "elec_margin_volume.png"), p1, width = 8, height = 5, dpi = 150)
file.copy(file.path(FIGDIR, "elec_margin_volume.png"),
          file.path(dirname(OUTDIR), "docs", "output", "figures", "elec_margin_volume.png"),
          overwrite = TRUE)
cat("  Saved elec_margin_volume.png\n")
#}

# =============================================================================
# SECTION 5: Figure 2 — Vocabulary: safe vs marginal
# =============================================================================
#{
cat("[I1] Figure 2: vocabulary safe vs marginal...\n")

word_counts <- starred_with_margin %>%
  filter(seat_type %in% c("Safe", "Marginal")) %>%
  unnest_tokens(word, question_text) %>%
  filter(
    !word %in% COMBINED_STOP,
    str_detect(word, "^[a-z]{5,}$")
  ) %>%
  count(seat_type, word) %>%
  group_by(word) %>%
  filter(sum(n) >= 20) %>%
  ungroup() %>%
  pivot_wider(names_from = seat_type, values_from = n, values_fill = 0) %>%
  mutate(
    safe_rate     = (Safe + 0.5) / (sum(Safe) + 0.5),
    marginal_rate = (Marginal + 0.5) / (sum(Marginal) + 0.5),
    log_ratio     = log2(safe_rate / marginal_rate)
  )

top_safe     <- word_counts %>% arrange(desc(log_ratio)) %>% head(15)
top_marginal <- word_counts %>% arrange(log_ratio)       %>% head(15)
plot_words   <- bind_rows(top_safe, top_marginal) %>%
  mutate(
    direction = if_else(log_ratio > 0, "Safe seat MPs", "Marginal seat MPs"),
    word      = reorder(word, log_ratio)
  )

p2 <- ggplot(plot_words, aes(x = log_ratio, y = word, fill = direction)) +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_fill_manual(values = c("Safe seat MPs" = "#1a9641", "Marginal seat MPs" = "#d7191c")) +
  labs(
    title    = "Safe seat MPs ask more policy questions; marginal seat MPs focus on local schemes",
    subtitle = paste0("Top 15 words by log2 ratio: safe (>", SAFE_THRESH*100, "% margin) vs marginal (<", MARGINAL_THRESH*100, "% margin) seat MPs"),
    x        = "Log2 ratio (positive = safe seat MPs use more)",
    y        = NULL,
    fill     = NULL,
    caption  = "Filtered to words appearing 20+ times total. Stop words, names, fragments removed."
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 11), legend.position = "bottom")

ggsave(file.path(FIGDIR, "elec_margin_vocab.png"), p2, width = 8, height = 6, dpi = 150)
file.copy(file.path(FIGDIR, "elec_margin_vocab.png"),
          file.path(dirname(OUTDIR), "docs", "output", "figures", "elec_margin_vocab.png"),
          overwrite = TRUE)
cat("  Saved elec_margin_vocab.png\n")
#}

# =============================================================================
# SECTION 6: Figure 3 — Temporal spike near dissolution
# =============================================================================
#{
cat("[I1] Figure 3: temporal spike...\n")

# Session 1-16 within the 16th LS; mark last 3 sessions as pre-election
temporal <- starred %>%
  filter(lok_no == 16L) %>%
  inner_join(
    mp_data %>% filter(lok_no == 16L) %>% select(mp_norm, seat_type),
    by = c("primary_norm" = "mp_norm")
  ) %>%
  mutate(session_no = as.integer(session_no)) %>%
  filter(!is.na(session_no)) %>%
  count(session_no, seat_type) %>%
  group_by(seat_type) %>%
  mutate(n_norm = n / max(n)) %>%
  ungroup()

n_sessions <- max(temporal$session_no, na.rm = TRUE)
pre_elect_cutoff <- n_sessions - 2

p3 <- ggplot(temporal, aes(x = session_no, y = n_norm, color = seat_type)) +
  annotate("rect", xmin = pre_elect_cutoff - 0.5, xmax = n_sessions + 0.5,
           ymin = 0, ymax = Inf, fill = "grey90", alpha = 0.5) +
  annotate("text", x = pre_elect_cutoff, y = 0.98,
           label = "Pre-dissolution", hjust = 0, size = 3.2, color = "grey40") +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.5) +
  scale_color_manual(values = c(
    "Safe" = "#1a9641", "Marginal" = "#d7191c", "Competitive" = "#f1a340"
  )) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_x_continuous(breaks = seq(1, n_sessions, 2)) +
  labs(
    title    = "All MP groups increase activity in final sessions",
    subtitle = "Normalised question count by session, 16th Lok Sabha (2014-2019)",
    x        = "Session number",
    y        = "Relative question volume (normalised to group max)",
    color    = "Seat type",
    caption  = "Shaded area = last 3 sessions before dissolution."
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")

ggsave(file.path(FIGDIR, "elec_temporal_spike.png"), p3, width = 8, height = 5, dpi = 150)
file.copy(file.path(FIGDIR, "elec_temporal_spike.png"),
          file.path(dirname(OUTDIR), "docs", "output", "figures", "elec_temporal_spike.png"),
          overwrite = TRUE)
cat("  Saved elec_temporal_spike.png\n")
#}

# =============================================================================
# SECTION 7: Figure 4 — Ministry targeting by seat type
# =============================================================================
#{
cat("[I1] Figure 4: ministry targeting...\n")

min_counts <- starred_with_margin %>%
  filter(seat_type %in% c("Safe", "Marginal"), !is.na(ministry), ministry != "") %>%
  mutate(ministry = str_to_title(str_trunc(ministry, 35))) %>%
  count(seat_type, ministry) %>%
  group_by(seat_type) %>%
  mutate(share = n / sum(n)) %>%
  ungroup()

min_wide <- min_counts %>%
  select(seat_type, ministry, share) %>%
  pivot_wider(names_from = seat_type, values_from = share, values_fill = 0) %>%
  mutate(diff = Safe - Marginal)

top_min <- bind_rows(
  min_wide %>% arrange(desc(diff)) %>% head(10),
  min_wide %>% arrange(diff)       %>% head(10)
) %>%
  distinct() %>%
  mutate(
    direction = if_else(diff > 0, "Safe seat MPs over-index", "Marginal seat MPs over-index"),
    ministry  = reorder(ministry, diff)
  )

p4 <- ggplot(top_min, aes(x = diff * 100, y = ministry, fill = direction)) +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  scale_fill_manual(
    values = c("Safe seat MPs over-index" = "#1a9641",
               "Marginal seat MPs over-index" = "#d7191c")
  ) +
  scale_x_continuous(labels = function(x) paste0(x, " pp")) +
  labs(
    title    = "What ministries do marginal vs safe seat MPs target?",
    subtitle = "Percentage point difference in ministry share: safe minus marginal seat MPs",
    x        = "Difference in ministry share (percentage points)",
    y        = NULL,
    fill     = NULL,
    caption  = "Ministry share = pct of each group's starred questions directed at that ministry."
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 11), legend.position = "bottom")

ggsave(file.path(FIGDIR, "elec_safe_vs_marginal.png"), p4, width = 9, height = 7, dpi = 150)
file.copy(file.path(FIGDIR, "elec_safe_vs_marginal.png"),
          file.path(dirname(OUTDIR), "docs", "output", "figures", "elec_safe_vs_marginal.png"),
          overwrite = TRUE)
cat("  Saved elec_safe_vs_marginal.png\n")
#}

# =============================================================================
# SECTION 8: Save summary
# =============================================================================
#{
summary_out <- mp_data %>%
  group_by(seat_type, lok_no) %>%
  summarise(
    n_mps         = n(),
    avg_margin    = round(mean(margin_pct), 3),
    avg_questions = round(mean(n_questions), 1),
    med_questions = median(n_questions),
    .groups = "drop"
  )

write_csv(summary_out, file.path(TABDIR, "electoral_scores.csv"))
cat("[I1] Done. Saved electoral_scores.csv\n\n")
print(summary_out)
#}
