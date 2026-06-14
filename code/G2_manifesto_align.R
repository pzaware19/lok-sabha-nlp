# =============================================================================
# G2_manifesto_align.R — Manifesto vs Parliamentary Questions Alignment
# Author: Piyush Zaware
# Updated: 2026-06-14
#
# PURPOSE:
#   Compares what parties PROMISE in election manifestos to what their MPs
#   actually QUESTION in parliament. Key question: do ruling parties follow
#   through on their manifesto agenda in parliament? Do opposition parties
#   hold the government accountable for its own promises?
#
#   Method: TF-IDF cosine similarity between manifesto vocabulary and
#   question vocabulary, computed per (party × Lok Sabha).
#   Also computes "promise gaps" — words prominent in a manifesto but
#   absent from that party's parliamentary questions.
#
# INPUTS:
#   output/tables/manifesto_text.csv        (from G1)
#   tmp/train-*.parquet
#   input/mp_party_lookup.csv
#
# OUTPUTS:
#   output/figures/manifesto_alignment_heatmap.png
#   output/figures/manifesto_bjp_promise_gap.png
#   output/figures/manifesto_opposition_accountability.png
#   output/figures/manifesto_top_words_comparison.png
#   output/tables/manifesto_alignment_scores.csv
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

pkgs <- c("arrow","tidyverse","tidytext","patchwork","ggrepel","stopwords","scales")
to_inst <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_inst) > 0) install.packages(to_inst)
suppressPackageStartupMessages(lapply(pkgs, library, character.only = TRUE))

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")

SAFFRON <- "#FF6B35"
NAVY    <- "#0D1B2A"
GREEN   <- "#138808"
PURPLE  <- "#6A3D9A"
TEAL    <- "#2CA25F"
BJP_COL <- "#FF9933"
INC_COL <- "#19AAED"

# ============================================================
# SECTION 1: Load and prepare manifesto text
# ============================================================
#{
cat("Loading manifesto text...\n")
manifesto_raw <- read_csv(file.path(TABDIR, "manifesto_text.csv"),
                           show_col_types = FALSE) %>%
  filter(lok_no >= 16, n_chars > 5000)  # drop 15th LS + near-empty

# Normalize party names to match questions data
party_recode_man <- c(
  "BJP"    = "BJP",
  "INC"    = "INC",
  "AITC"   = "AITC",
  "NCP"    = "NCP",
  "AIADMK" = "AIADMK",
  "CPI-M"  = "CPI(M)",
  "DMK"    = "DMK"
)
manifesto_raw <- manifesto_raw %>%
  mutate(party_q = recode(party, !!!party_recode_man, .default = party))

cat(sprintf("  %d manifesto docs across %d parties and Lok Sabhas 16-18\n",
            nrow(manifesto_raw), n_distinct(manifesto_raw$party)))
#}

# ============================================================
# SECTION 2: Load starred questions + party labels
# ============================================================
#{
cat("Loading starred questions...\n")
parquet_files <- list.files(TMPDIR, pattern = "train-.*\\.parquet$", full.names = TRUE)
raw <- purrr::map_dfr(parquet_files, function(f)
  read_parquet(f, col_select = c("id","lok_no","type","ministry","members","question_text")))

starred <- raw %>% filter(type == "STARRED", lok_no >= 16)

get_primary <- function(x) {
  tryCatch({
    items <- as.character(list(x)[[1]])
    str_squish(items[1])
  }, error = function(e) NA_character_)
}
starred <- starred %>%
  mutate(primary_raw = map_chr(members, get_primary))

# Name normaliser
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
    mp_key      = str_to_upper(str_squish(mp_name)),
    mp_stripped = vapply(mp_key, strip_hon, character(1)),
    mp_norm     = vapply(mp_stripped, norm_fl, character(1))
  ) %>%
  arrange(desc(lok_no)) %>%
  distinct(mp_norm, .keep_all = TRUE)

mp_party <- setNames(lookup$party_family, lookup$mp_norm)

starred <- starred %>%
  mutate(
    primary_norm = vapply(vapply(primary_raw, strip_hon, character(1)), norm_fl, character(1)),
    party_family = mp_party[primary_norm]
  ) %>%
  filter(!is.na(party_family), !is.na(question_text))

cat(sprintf("  Starred (LS 16-18, party-matched): %d questions\n", nrow(starred)))
#}

# ============================================================
# SECTION 3: Build word-frequency matrices
# ============================================================
#{
cat("Tokenising...\n")

# Load shared stop words + MP name blocklist (builds COMBINED_STOP)
source(file.path(CODDIR, "._stop_words.R"))

# Question word counts per (party_family, lok_no) — built first so we can
# use the question vocabulary as a filter for manifesto words.
q_words <- starred %>%
  select(party_family, lok_no, question_text) %>%
  unnest_tokens(word, question_text) %>%
  filter(!word %in% COMBINED_STOP,
         str_detect(word, "^[a-z]+$"), nchar(word) >= 5) %>%
  count(party_family, lok_no, word, name = "n_q")

# Valid word set from questions corpus (structured text, no OCR artifacts).
# Filtering manifesto tokens here removes PDF hyphenation fragments and
# Hindi transliterations that don't appear in parliamentary question text.
q_valid_words <- q_words %>% distinct(word) %>% pull(word)

# Manifesto word counts per (party, lok_no)
man_words <- manifesto_raw %>%
  select(party_q, lok_no, text) %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% COMBINED_STOP,
         str_detect(word, "^[a-z]+$"), nchar(word) >= 5,
         word %in% q_valid_words) %>%
  count(party_q, lok_no, word, name = "n_man")

cat(sprintf("  Manifesto vocab (clean): %d unique words\n", n_distinct(man_words$word)))
cat(sprintf("  Question vocab (clean):  %d unique words\n", n_distinct(q_words$word)))
#}

# ============================================================
# SECTION 4: Cosine similarity (TF-IDF based)
# ============================================================
#{
cat("Computing TF-IDF cosine similarities...\n")

# TF-IDF for manifestos
man_tfidf <- man_words %>%
  mutate(doc_id = paste(party_q, lok_no, sep = "_")) %>%
  bind_tf_idf(word, doc_id, n_man)

# TF-IDF for questions
q_tfidf <- q_words %>%
  mutate(doc_id = paste(party_family, lok_no, sep = "_")) %>%
  bind_tf_idf(word, doc_id, n_q)

# Cosine similarity function
cosine_sim <- function(a, b) {
  shared <- intersect(names(a), names(b))
  if (length(shared) == 0) return(0)
  num   <- sum(a[shared] * b[shared])
  denom <- sqrt(sum(a^2)) * sqrt(sum(b^2))
  if (denom == 0) return(0)
  num / denom
}

# Compute alignment for each (party, lok_no) pair
parties_to_compare <- manifesto_raw %>%
  distinct(party_q, lok_no) %>%
  rename(party = party_q)

alignment_rows <- purrr::pmap_dfr(parties_to_compare, function(party, lok_no) {
  # Get question party name (same as manifesto party name after recode)
  man_vec <- man_tfidf %>%
    filter(party_q == party, lok_no == .env$lok_no) %>%
    { setNames(.$tf_idf, .$word) }

  q_vec   <- q_tfidf %>%
    filter(party_family == party, lok_no == .env$lok_no) %>%
    { setNames(.$tf_idf, .$word) }

  n_q_docs <- sum(q_words$n_q[q_words$party_family == party &
                                 q_words$lok_no    == lok_no], na.rm = TRUE)

  tibble(
    party   = party,
    lok_no  = lok_no,
    cosine  = cosine_sim(man_vec, q_vec),
    n_man_words = length(man_vec),
    n_q_words   = length(q_vec),
    n_q_total   = n_q_docs
  )
}) %>%
  filter(n_q_words > 0)  # drop party × LS with no questions

write_csv(alignment_rows, file.path(TABDIR, "manifesto_alignment_scores.csv"))
cat("  Alignment scores computed.\n")
print(alignment_rows %>% arrange(party, lok_no) %>% select(party, lok_no, cosine, n_q_total))
#}

# ============================================================
# SECTION 5: Figure 1 — Alignment heatmap
# ============================================================
#{
cat("Plotting alignment heatmap...\n")

# Only include parties with data in at least 2 cells
parties_ok <- alignment_rows %>%
  count(party) %>%
  filter(n >= 2) %>%
  pull(party)

plot_data <- alignment_rows %>%
  filter(party %in% parties_ok) %>%
  mutate(
    ls_label   = paste0(lok_no, "th LS\n(",
                        case_when(lok_no == 16 ~ "2014-19",
                                  lok_no == 17 ~ "2019-24",
                                  lok_no == 18 ~ "2024-"),
                        ")"),
    party_role = case_when(
      party == "BJP"  ~ "BJP (Ruling 2014-)",
      party == "INC"  ~ "INC (Opposition)",
      TRUE            ~ party
    ),
    party_role = fct_reorder(party_role, cosine, .fun = mean, .desc = TRUE)
  )

p_heatmap <- ggplot(plot_data,
                    aes(x = ls_label, y = party_role, fill = cosine)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.3f\n(%d Qs)", cosine, n_q_total)),
            size = 3.0, colour = "white", fontface = "bold") +
  scale_fill_gradient2(
    low      = "#F5F0FF",
    mid      = "#9B59B6",
    high     = "#4A0E8F",
    midpoint = 0.15,
    name     = "Cosine\nsimilarity",
    limits   = c(0, 0.35),
    oob      = scales::squish
  ) +
  labs(
    title    = "How closely do parties question what they promise?",
    subtitle = "TF-IDF cosine similarity between each party's election manifesto\nand its MPs' starred questions in the subsequent Lok Sabha.",
    x = NULL, y = NULL,
    caption  = "Higher = manifesto vocabulary and question vocabulary are more aligned."
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", colour = NAVY, size = 14),
    plot.subtitle = element_text(colour = "grey40", size = 10),
    plot.caption  = element_text(colour = "grey60", size = 8),
    axis.text     = element_text(size = 10),
    legend.position = "right",
    panel.grid    = element_blank()
  )

ggsave(file.path(FIGDIR, "manifesto_alignment_heatmap.png"),
       p_heatmap, width = 10, height = 6, dpi = 180)
cat("Saved: manifesto_alignment_heatmap.png\n")
#}

# ============================================================
# SECTION 6: Figure 2 — BJP promise gap across Lok Sabhas
# ============================================================
#{
cat("Plotting BJP promise gap...\n")

bjp_man <- man_tfidf %>%
  filter(party_q == "BJP") %>%
  group_by(word) %>%
  summarise(man_tfidf = mean(tf_idf), .groups = "drop")   # average across years

bjp_q <- q_tfidf %>%
  filter(party_family == "BJP") %>%
  group_by(word) %>%
  summarise(q_tfidf = mean(tf_idf), .groups = "drop")

bjp_gap <- full_join(bjp_man, bjp_q, by = "word") %>%
  replace_na(list(man_tfidf = 0, q_tfidf = 0)) %>%
  mutate(
    gap       = man_tfidf - q_tfidf,
    direction = case_when(
      gap >  0.0002 ~ "Promised but NOT questioned",
      gap < -0.0002 ~ "Questioned beyond manifesto",
      TRUE          ~ "Aligned"
    )
  ) %>%
  filter(man_tfidf + q_tfidf > 0.0001)   # remove zero-zero noise

# Top promised-but-not-questioned + top questioned-beyond-manifesto
top_gap <- bind_rows(
  bjp_gap %>% filter(direction == "Promised but NOT questioned") %>% slice_max(gap, n = 18),
  bjp_gap %>% filter(direction == "Questioned beyond manifesto")  %>% slice_min(gap, n = 12)
) %>%
  mutate(word = fct_reorder(word, gap))

p_bjp_gap <- ggplot(top_gap, aes(x = gap, y = word, fill = direction)) +
  geom_col(width = 0.75) +
  geom_vline(xintercept = 0, linewidth = 0.5, colour = "grey40") +
  scale_fill_manual(
    values = c(
      "Promised but NOT questioned"  = BJP_COL,
      "Questioned beyond manifesto"  = NAVY,
      "Aligned"                      = "grey70"
    ),
    name = NULL
  ) +
  labs(
    title    = "BJP: what they promised vs what they questioned",
    subtitle = "Orange = prominent in BJP manifestos but absent from BJP MPs' starred questions.\nNavy = topics BJP MPs questioned that barely appear in their manifestos.",
    x = "TF-IDF gap (manifesto minus questions)", y = NULL,
    caption  = "Averaged across 16th, 17th and 18th Lok Sabha."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", colour = NAVY),
    plot.subtitle = element_text(colour = "grey40", size = 9),
    legend.position = "bottom"
  )

ggsave(file.path(FIGDIR, "manifesto_bjp_promise_gap.png"),
       p_bjp_gap, width = 10, height = 8, dpi = 180)
cat("Saved: manifesto_bjp_promise_gap.png\n")
#}

# ============================================================
# SECTION 7: Figure 3 — Opposition accountability
# Does INC question BJP about BJP's own manifesto promises?
# ============================================================
#{
cat("Plotting opposition accountability...\n")

inc_q_tfidf <- q_tfidf %>%
  filter(party_family == "INC") %>%
  group_by(word) %>%
  summarise(inc_q_tfidf = mean(tf_idf), .groups = "drop")

inc_accountability <- full_join(bjp_man, inc_q_tfidf, by = "word") %>%
  replace_na(list(man_tfidf = 0, inc_q_tfidf = 0)) %>%
  mutate(
    accountability = pmin(man_tfidf, inc_q_tfidf) /
                     pmax(man_tfidf + inc_q_tfidf, 1e-8),
    in_bjp_man     = man_tfidf > 0.0001,
    in_inc_q       = inc_q_tfidf > 0.0001,
    overlap        = in_bjp_man & in_inc_q
  )

# Words in BJP manifesto that INC MPs question about (accountability) vs not
acc_long <- bind_rows(
  inc_accountability %>%
    filter(in_bjp_man) %>%
    slice_max(inc_q_tfidf, n = 20) %>%
    mutate(panel = "BJP promise words\nINC MPs DO question"),
  inc_accountability %>%
    filter(in_bjp_man, !in_inc_q) %>%
    slice_max(man_tfidf, n = 16) %>%
    mutate(panel = "BJP promise words\nINC MPs DON'T question")
) %>%
  mutate(word = fct_reorder(word, inc_q_tfidf + man_tfidf))

p_accountability <- acc_long %>%
  ggplot(aes(x = inc_q_tfidf + man_tfidf,
             y = fct_reorder(word, inc_q_tfidf + man_tfidf),
             fill = panel)) +
  geom_col(width = 0.75) +
  facet_wrap(~panel, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = c(
    "BJP promise words\nINC MPs DO question"    = INC_COL,
    "BJP promise words\nINC MPs DON'T question" = "grey60"
  ), guide = "none") +
  labs(
    title    = "Opposition accountability: does INC question BJP on its own promises?",
    subtitle = "Left: BJP manifesto words that INC MPs actively pick up in starred questions.\nRight: BJP promises that INC MPs largely ignore.",
    x = "Combined TF-IDF weight", y = NULL,
    caption  = "INC questions from 16th, 17th and 18th Lok Sabha; BJP manifestos 2014-2024."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", colour = NAVY),
    plot.subtitle = element_text(colour = "grey40", size = 9),
    strip.text    = element_text(face = "bold", colour = NAVY)
  )

ggsave(file.path(FIGDIR, "manifesto_opposition_accountability.png"),
       p_accountability, width = 12, height = 7, dpi = 180)
cat("Saved: manifesto_opposition_accountability.png\n")
#}

# ============================================================
# SECTION 8: Figure 4 — BJP manifesto drift across Lok Sabhas
# 2014 manifesto vs BJP questions in 16th, 17th, 18th LS
# ============================================================
#{
cat("Plotting BJP manifesto drift...\n")

bjp_man_2014 <- man_tfidf %>%
  filter(party_q == "BJP", lok_no == 16) %>%
  select(word, man_score = tf_idf)

bjp_q_by_ls <- q_tfidf %>%
  filter(party_family == "BJP") %>%
  select(lok_no, word, q_score = tf_idf)

# Track top BJP 2014 manifesto words across Lok Sabhas
bjp_man_top <- bjp_man_2014 %>% slice_max(man_score, n = 25) %>% pull(word)

bjp_word_track <- bjp_q_by_ls %>%
  filter(word %in% bjp_man_top) %>%
  group_by(lok_no) %>%
  mutate(q_rank = rank(-q_score)) %>%
  ungroup() %>%
  mutate(ls_label = paste0(lok_no, "th LS"))

p_drift <- ggplot(bjp_word_track,
                  aes(x = ls_label, y = q_score,
                      group = word, colour = word)) +
  geom_line(linewidth = 0.8, alpha = 0.7) +
  geom_point(size = 2, alpha = 0.8) +
  ggrepel::geom_text_repel(
    data = bjp_word_track %>% filter(lok_no == max(lok_no)),
    aes(label = word), size = 3, hjust = -0.1,
    direction = "y", max.overlaps = 15, segment.size = 0.3
  ) +
  scale_colour_viridis_d(guide = "none") +
  scale_x_discrete(expand = expansion(add = c(0.3, 1.2))) +
  labs(
    title    = "Did BJP MPs keep questioning their 2014 manifesto agenda?",
    subtitle = "Tracking BJP's top 2014 manifesto keywords in their starred questions\nacross three Lok Sabhas. Rising = gaining traction; falling = fading agenda.",
    x = NULL, y = "TF-IDF weight in BJP questions",
    caption  = "Vocabulary from BJP 2014 manifesto; question data from 16th, 17th, 18th LS."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", colour = NAVY),
    plot.subtitle = element_text(colour = "grey40", size = 9)
  )

ggsave(file.path(FIGDIR, "manifesto_bjp_drift.png"),
       p_drift, width = 11, height = 7, dpi = 180)
cat("Saved: manifesto_bjp_drift.png\n")
#}

# ============================================================
# SECTION 9: Figure 5 — Side-by-side top-20 words: manifesto vs questions
# For BJP and INC
# ============================================================
#{
cat("Plotting top-word comparison...\n")

build_comparison <- function(party_name, party_colour) {
  man_top <- man_tfidf %>%
    filter(party_q == party_name) %>%
    group_by(word) %>%
    summarise(score = mean(tf_idf), .groups = "drop") %>%
    slice_max(score, n = 20) %>%
    mutate(source = "Manifesto promises", party = party_name)

  q_top <- q_tfidf %>%
    filter(party_family == party_name) %>%
    group_by(word) %>%
    summarise(score = mean(tf_idf), .groups = "drop") %>%
    slice_max(score, n = 20) %>%
    mutate(source = "Parliamentary questions", party = party_name)

  bind_rows(man_top, q_top)
}

compare_bjp <- build_comparison("BJP", BJP_COL)
compare_inc <- build_comparison("INC", INC_COL)

comparison_all <- bind_rows(compare_bjp, compare_inc) %>%
  mutate(
    panel = paste0(party, " — ", source),
    word  = fct_reorder(word, score)
  )

# Shared words (both in manifesto AND questions) highlighted
shared_bjp <- intersect(compare_bjp$word[compare_bjp$source == "Manifesto promises"],
                         compare_bjp$word[compare_bjp$source == "Parliamentary questions"])
shared_inc <- intersect(compare_inc$word[compare_inc$source == "Manifesto promises"],
                         compare_inc$word[compare_inc$source == "Parliamentary questions"])

comparison_all <- comparison_all %>%
  mutate(shared = case_when(
    party == "BJP" & word %in% shared_bjp ~ TRUE,
    party == "INC" & word %in% shared_inc ~ TRUE,
    TRUE ~ FALSE
  ))

p_comparison <- comparison_all %>%
  ggplot(aes(x = score, y = word,
             fill  = ifelse(shared, "Appears in both", source),
             alpha = ifelse(shared, 1, 0.8))) +
  geom_col(width = 0.75) +
  facet_wrap(~panel, scales = "free", ncol = 4) +
  scale_fill_manual(
    values = c(
      "Manifesto promises"     = "grey70",
      "Parliamentary questions"= "grey40",
      "Appears in both"        = GREEN
    ),
    name = NULL
  ) +
  scale_alpha_identity() +
  labs(
    title    = "Manifesto vocabulary vs parliamentary question vocabulary",
    subtitle = "Green bars appear in BOTH the party's manifesto top-20 AND their questions top-20.\nGrey = appears in only one. Overlap reveals how much manifesto language bleeds into parliament.",
    x = "Mean TF-IDF score", y = NULL,
    caption  = "Top 20 distinctive words per party × source. Common parliamentary boilerplate removed."
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title      = element_text(face = "bold", colour = NAVY, size = 13),
    plot.subtitle   = element_text(colour = "grey40", size = 9),
    strip.text      = element_text(face = "bold", size = 9),
    legend.position = "bottom",
    panel.grid.major.y = element_blank()
  )

ggsave(file.path(FIGDIR, "manifesto_top_words_comparison.png"),
       p_comparison, width = 14, height = 8, dpi = 180)
cat("Saved: manifesto_top_words_comparison.png\n")
#}

cat("\n=== G2 complete ===\n")
cat(sprintf("Alignment scores saved to: %s\n", file.path(TABDIR, "manifesto_alignment_scores.csv")))
