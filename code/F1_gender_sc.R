# =============================================================================
# F1_gender_sc.R — Gender and SC/ST Analysis of Lok Sabha Starred Questions
# Author: Piyush Zaware
# Updated: 2026-06-13
#
# PURPOSE:
#   Analyses whether gender and caste (SC/ST reserved seat status) shape
#   parliamentary questioning behaviour — ministry targeting, tone, vocabulary,
#   and party representation.
#
#   Gender identification: honorific prefixes in parquet member names
#   (SHRIMATI, SMT, KUMARI, MRS, MS). These are the official parliamentary
#   forms of address for women MPs and are reliable identifiers.
#
#   SC/ST identification: constituency column in mp_party_lookup.csv contains
#   (SC) and (ST) tags for reserved seats, which is a clean proxy.
#
# INPUTS:
#   tmp/train-*.parquet
#   input/mp_party_lookup.csv
#   output/tables/sentiment_doc.csv
#   output/tables/bertopic_doc_assignments.csv
#   output/tables/bertopic_topic_words.csv
#
# OUTPUTS:
#   output/figures/gender_*.png
#   output/figures/scst_*.png
#   output/tables/gender_ministry.csv
#   output/tables/scst_ministry.csv
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

pkgs <- c("arrow","tidyverse","tidytext","patchwork","ggrepel","stopwords")
to_inst <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_inst) > 0) install.packages(to_inst)
suppressPackageStartupMessages(lapply(pkgs, library, character.only = TRUE))

FIGDIR  <- file.path(OUTDIR, "figures")
TABDIR  <- file.path(OUTDIR, "tables")
SAFFRON <- "#FF6B35"
NAVY    <- "#0D1B2A"
GREEN   <- "#138808"
PURPLE  <- "#6A3D9A"
TEAL    <- "#2CA25F"

# ============================================================
# SECTION 1: Load data and tag gender + SC/ST
# ============================================================
#{
cat("Loading parquet files...\n")
parquet_files <- list.files(TMPDIR, pattern = "train-.*\\.parquet$",
                             full.names = TRUE)
raw <- purrr::map_dfr(parquet_files, function(f)
  read_parquet(f, col_select = c("id","lok_no","session_no","type",
                                  "ministry","members","question_text")))

starred <- raw %>% filter(type == "STARRED")
cat(sprintf("  Total starred: %d\n", nrow(starred)))

# Extract primary member name
get_primary <- function(x) {
  tryCatch({
    items <- as.character(list(x)[[1]])
    str_squish(items[1])
  }, error = function(e) NA_character_)
}
starred <- starred %>%
  mutate(primary_raw = map_chr(members, get_primary))

# ── Gender: honorific-based ────────────────────────────────────────────────
FEMALE_PAT <- regex("\\b(SHRIMATI|SMT\\.?|KUMARI|MRS\\.?|MS\\.?)\\b",
                     ignore_case = TRUE)
starred <- starred %>%
  mutate(is_female = str_detect(str_to_upper(primary_raw), FEMALE_PAT))

n_female <- sum(starred$is_female, na.rm = TRUE)
cat(sprintf("  Female MP questions (honorific): %d (%.1f%%)\n",
            n_female, 100 * n_female / nrow(starred)))

# ── Load party lookup ───────────────────────────────────────────────────────
lookup <- read_csv(file.path(INPDIR, "mp_party_lookup.csv"),
                   show_col_types = FALSE) %>%
  mutate(mp_key = str_to_upper(str_squish(mp_name)))

# Strip honorifics for matching
strip_honorific <- function(s) {
  s <- str_to_upper(str_squish(s))
  s <- str_remove_all(s, "\\b(SHRIMATI|SMT\\.?|KUMARI|MRS\\.?|MS\\.?|DR\\.?|PROF\\.?|SH\\.?|SHRI\\.?)\\b")
  str_squish(s)
}
norm_fl <- function(s) {
  parts <- str_split(str_squish(s), "\\s+")[[1]]
  if (length(parts) <= 2) return(s)
  paste(parts[1], parts[length(parts)])
}

lookup <- lookup %>%
  mutate(
    mp_key_stripped = vapply(mp_key, strip_honorific, character(1)),
    mp_key_norm     = vapply(mp_key_stripped, norm_fl, character(1)),
    sc = str_detect(constituency, regex("\\(SC\\)", ignore_case = TRUE)),
    st = str_detect(constituency, regex("\\(ST\\)", ignore_case = TRUE))
  ) %>%
  arrange(desc(lok_no)) %>%
  distinct(mp_key_norm, .keep_all = TRUE)

mp_to_party <- setNames(lookup$party_family, lookup$mp_key_norm)
mp_to_sc    <- setNames(lookup$sc, lookup$mp_key_norm)
mp_to_st    <- setNames(lookup$st, lookup$mp_key_norm)
mp_to_const <- setNames(lookup$constituency, lookup$mp_key_norm)

cat(sprintf("  SC constituencies: %d MPs\n", sum(lookup$sc, na.rm=TRUE)))
cat(sprintf("  ST constituencies: %d MPs\n", sum(lookup$st, na.rm=TRUE)))

# Match starred → party / SC / ST
starred <- starred %>%
  mutate(
    primary_stripped = vapply(primary_raw, strip_honorific, character(1)),
    primary_norm     = vapply(primary_stripped, norm_fl, character(1)),
    party_family     = mp_to_party[primary_norm],
    is_sc            = coalesce(mp_to_sc[primary_norm], FALSE),
    is_st            = coalesce(mp_to_st[primary_norm], FALSE),
    seat_type        = case_when(
      is_sc ~ "SC Reserved",
      is_st ~ "ST Reserved",
      TRUE  ~ "General"
    )
  )

cat(sprintf("  Party-matched: %d (%.1f%%)\n",
            sum(!is.na(starred$party_family)), 100*mean(!is.na(starred$party_family))))
cat(sprintf("  SC MP questions: %d\n", sum(starred$is_sc, na.rm=TRUE)))
cat(sprintf("  ST MP questions: %d\n", sum(starred$is_st, na.rm=TRUE)))

# Ministry cleaning
ministry_recode <- c(
  "AGRICULTURE AND FARMERS WELFARE" = "Agriculture",
  "AGRICULTURE"                     = "Agriculture",
  "RAILWAYS"                        = "Railways",
  "FINANCE"                         = "Finance",
  "HOME AFFAIRS"                    = "Home Affairs",
  "HEALTH AND FAMILY WELFARE"       = "Health",
  "HEALTH"                          = "Health",
  "COMMUNICATIONS"                  = "Communications",
  "EDUCATION"                       = "Education",
  "HUMAN RESOURCE DEVELOPMENT"      = "Education",
  "ROAD TRANSPORT AND HIGHWAYS"     = "Road Transport",
  "COMMERCE AND INDUSTRY"           = "Commerce",
  "SOCIAL JUSTICE AND EMPOWERMENT"  = "Social Justice",
  "POWER"                           = "Power",
  "EXTERNAL AFFAIRS"                = "External Affairs",
  "HOUSING AND URBAN AFFAIRS"       = "Urban Development",
  "WOMEN AND CHILD DEVELOPMENT"     = "Women & Child",
  "LABOUR AND EMPLOYMENT"           = "Labour",
  "RURAL DEVELOPMENT"               = "Rural Development",
  "DEFENCE"                         = "Defence",
  "PETROLEUM AND NATURAL GAS"       = "Petroleum",
  "JAL SHAKTI"                      = "Jal Shakti",
  "WATER RESOURCES"                 = "Jal Shakti",
  "TRIBAL AFFAIRS"                  = "Tribal Affairs",
  "STEEL"                           = "Steel",
  "TEXTILES"                        = "Textiles",
  "MINORITY AFFAIRS"                = "Minority Affairs",
  "SCIENCE AND TECHNOLOGY"          = "Science & Tech",
  "INFORMATION AND BROADCASTING"    = "I&B",
  "COAL"                            = "Coal",
  "MINES"                           = "Mines",
  "PANCHAYATI RAJ"                  = "Panchayati Raj",
  "MICRO,SMALL AND MEDIUM ENTERPRISES" = "MSME",
  "CONSUMER AFFAIRS, FOOD AND PUBLIC DISTRIBUTION" = "Food & Consumer"
)

starred <- starred %>%
  mutate(ministry_clean = recode(str_to_upper(str_trim(ministry)),
                                  !!!ministry_recode,
                                  .default = str_to_title(str_trim(ministry))))
#}

# ============================================================
# SECTION 2: Gender — Ministry targeting
# ============================================================
#{

nat_min_share <- starred %>%
  count(ministry_clean, name = "nat_n") %>%
  mutate(nat_share = nat_n / sum(nat_n))

female_min <- starred %>%
  filter(is_female) %>%
  count(ministry_clean, name = "f_n") %>%
  mutate(f_share = f_n / sum(f_n))

male_min <- starred %>%
  filter(!is_female) %>%
  count(ministry_clean, name = "m_n") %>%
  mutate(m_share = m_n / sum(m_n))

gender_ministry <- female_min %>%
  left_join(male_min,    by = "ministry_clean") %>%
  left_join(nat_min_share, by = "ministry_clean") %>%
  mutate(
    gap          = f_share - m_share,
    pearson_f    = (coalesce(f_n, 0) - nrow(starred %>% filter(is_female)) * nat_share) /
                   sqrt(pmax(nrow(starred %>% filter(is_female)) * nat_share, 0.5)),
    f_pct        = 100 * coalesce(f_share, 0),
    m_pct        = 100 * coalesce(m_share, 0)
  ) %>%
  filter(coalesce(f_n, 0) >= 2) %>%
  arrange(desc(abs(gap)))

write_csv(gender_ministry, file.path(TABDIR, "gender_ministry.csv"))

# Divergence chart: female minus male share
p_gender_ministry <- gender_ministry %>%
  slice_max(abs(gap), n = 20) %>%
  mutate(ministry_clean = fct_reorder(ministry_clean, gap),
         direction = if_else(gap > 0, "Women ask more", "Men ask more")) %>%
  ggplot(aes(x = gap * 100, y = ministry_clean, fill = direction)) +
  geom_col(width = 0.75) +
  geom_vline(xintercept = 0, linewidth = 0.5, colour = "grey40") +
  scale_fill_manual(values = c("Women ask more" = PURPLE,
                               "Men ask more"   = TEAL),
                    name = NULL) +
  labs(
    title    = "Which ministries do women MPs question more?",
    subtitle = "Difference in share of questions (women minus men), percentage points.\nPositive = women disproportionately target this ministry.",
    x = "Percentage point difference (women − men)", y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", colour = NAVY),
    legend.position = "bottom"
  )

ggsave(file.path(FIGDIR, "gender_ministry.png"),
       p_gender_ministry, width = 10, height = 7, dpi = 180)
cat("Saved: gender_ministry.png\n")
#}

# ============================================================
# SECTION 3: Gender — Party representation
# ============================================================
#{

gender_party <- starred %>%
  filter(!is.na(party_family)) %>%
  group_by(party_family) %>%
  summarise(
    total    = n(),
    female_n = sum(is_female),
    female_pct = 100 * female_n / total,
    .groups = "drop"
  ) %>%
  filter(total >= 30) %>%
  arrange(desc(female_pct))

p_gender_party <- gender_party %>%
  mutate(party_family = fct_reorder(party_family, female_pct)) %>%
  ggplot(aes(x = female_pct, y = party_family, fill = female_pct)) +
  geom_col(width = 0.75) +
  geom_text(aes(label = sprintf("%.1f%%  (%d Qs)", female_pct, female_n)),
            hjust = -0.05, size = 3.2, colour = "grey30") +
  scale_fill_gradient(low = "#E8D5F0", high = PURPLE, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.35)),
                     labels = scales::percent_format(scale = 1)) +
  labs(
    title    = "Women's share of starred questions by party",
    subtitle = "% of each party's starred questions where the primary questioner is a woman MP.\n(Identified via parliamentary honorifics: Shrimati, Smt, Kumari, Ms, Mrs.)",
    x = "% of party's starred questions", y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", colour = NAVY))

ggsave(file.path(FIGDIR, "gender_party.png"),
       p_gender_party, width = 10, height = 6, dpi = 180)
cat("Saved: gender_party.png\n")
#}

# ============================================================
# SECTION 4: Gender — Tone (sentiment)
# ============================================================
#{
sent_path <- file.path(TABDIR, "sentiment_doc.csv")
if (file.exists(sent_path)) {
  sent_doc <- read_csv(sent_path, show_col_types = FALSE)

  starred_sent <- starred %>%
    select(id, is_female, party_family, lok_no) %>%
    inner_join(sent_doc %>% select(id, sent_compound), by = "id")

  gender_sent <- starred_sent %>%
    group_by(is_female) %>%
    summarise(
      n          = n(),
      mean_score = round(mean(sent_compound, na.rm=TRUE), 3),
      adv_rate   = round(100 * mean(sent_compound <= -0.05, na.rm=TRUE), 1),
      .groups    = "drop"
    ) %>%
    mutate(gender = if_else(is_female, "Women MPs", "Men MPs"))

  cat("\nGender sentiment comparison:\n")
  print(gender_sent)

  p_gender_tone <- ggplot(starred_sent,
                          aes(x = sent_compound,
                              fill = if_else(is_female, "Women MPs", "Men MPs"),
                              colour = if_else(is_female, "Women MPs", "Men MPs"))) +
    geom_density(alpha = 0.35, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    scale_fill_manual(values   = c("Women MPs" = PURPLE, "Men MPs" = TEAL),
                      name = NULL) +
    scale_colour_manual(values = c("Women MPs" = PURPLE, "Men MPs" = TEAL),
                        name = NULL) +
    annotate("text", x = gender_sent$mean_score[gender_sent$gender=="Women MPs"],
             y = 3.5, label = paste0("Women\nμ=", gender_sent$mean_score[gender_sent$gender=="Women MPs"]),
             colour = PURPLE, size = 3.2, fontface = "bold") +
    annotate("text", x = gender_sent$mean_score[gender_sent$gender=="Men MPs"],
             y = 3.5, label = paste0("Men\nμ=", gender_sent$mean_score[gender_sent$gender=="Men MPs"]),
             colour = TEAL, size = 3.2, fontface = "bold") +
    labs(
      title    = "Question tone by gender",
      subtitle = "VADER compound score. Left of 0 = adversarial framing.",
      x = "VADER compound score", y = "Density"
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold", colour = NAVY))

  ggsave(file.path(FIGDIR, "gender_tone.png"),
         p_gender_tone, width = 9, height = 5, dpi = 180)
  cat("Saved: gender_tone.png\n")
}
#}

# ============================================================
# SECTION 5: Gender — Distinctive vocabulary
# ============================================================
#{
clean_text <- function(t) {
  if (is.na(t)) return("")
  t <- str_replace_all(t, "##.*?\\n", " ")
  t <- str_replace_all(t, "\\([a-z]\\)", " ")
  str_squish(t)
}

custom_stop <- c("will","minister","whether","government","please",
                  "state","details","thereof","taken","steps","also",
                  "further","said","country","india","hon","aware",
                  "regard","provide","information","thereon","proposed",
                  "members","question","starred","unstarred","lok","sabha",
                  stopwords::stopwords("en"))

gender_corpus <- starred %>%
  mutate(
    text_c = map_chr(question_text, clean_text),
    doc    = if_else(is_female, "Women MPs", "Men MPs")
  ) %>%
  filter(nchar(text_c) > 30) %>%
  unnest_tokens(word, text_c) %>%
  filter(!word %in% custom_stop,
         str_detect(word, "^[a-z]+$"), nchar(word) >= 4) %>%
  count(doc, word) %>%
  bind_tf_idf(word, doc, n)

p_gender_vocab <- gender_corpus %>%
  filter(doc == "Women MPs") %>%
  slice_max(tf_idf, n = 20) %>%
  mutate(word = fct_reorder(word, tf_idf)) %>%
  ggplot(aes(x = word, y = tf_idf, fill = tf_idf)) +
  geom_col(width = 0.75, show.legend = FALSE) +
  coord_flip() +
  scale_fill_gradient(low = "#E8D5F0", high = PURPLE) +
  labs(
    title    = "Words most distinctive to women MPs' questions",
    subtitle = "TF-IDF vs all male MP questions combined.",
    x = NULL, y = "TF-IDF score"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", colour = NAVY),
        axis.text.y = element_text(size = 10))

ggsave(file.path(FIGDIR, "gender_vocab.png"),
       p_gender_vocab, width = 9, height = 7, dpi = 180)
cat("Saved: gender_vocab.png\n")
#}

# ============================================================
# SECTION 6: SC/ST — Ministry targeting
# ============================================================
#{

scst_min <- starred %>%
  group_by(seat_type, ministry_clean) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(seat_type) %>%
  mutate(share = 100 * n / sum(n)) %>%
  ungroup()

# Top ministries for each seat type
top_per_type <- scst_min %>%
  group_by(seat_type) %>%
  slice_max(share, n = 12) %>%
  ungroup() %>%
  pull(ministry_clean) %>%
  unique()

# Pivot and compute SC/ST excess over General
scst_wide <- scst_min %>%
  filter(ministry_clean %in% top_per_type) %>%
  pivot_wider(names_from = seat_type, values_from = c(n, share),
              values_fill = 0) %>%
  mutate(
    sc_excess = `share_SC Reserved` - share_General,
    st_excess = `share_ST Reserved` - share_General
  )

write_csv(scst_wide, file.path(TABDIR, "scst_ministry.csv"))

# SC excess chart
p_sc_ministry <- scst_wide %>%
  filter(abs(sc_excess) >= 0.3) %>%
  mutate(ministry_clean = fct_reorder(ministry_clean, sc_excess),
         direction = if_else(sc_excess > 0, "SC MPs ask more", "SC MPs ask less")) %>%
  ggplot(aes(x = sc_excess, y = ministry_clean, fill = direction)) +
  geom_col(width = 0.75) +
  geom_vline(xintercept = 0, linewidth = 0.5, colour = "grey40") +
  scale_fill_manual(values = c("SC MPs ask more" = "#E31A1C",
                               "SC MPs ask less" = "#AAAAAA"),
                    name = NULL) +
  labs(
    title    = "SC reserved-seat MPs: ministry focus vs general-seat MPs",
    subtitle = "Percentage point difference in share of questions. Positive = SC MPs focus more here.",
    x = "pp difference (SC minus General)", y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", colour = NAVY),
        legend.position = "bottom")

# ST excess chart
p_st_ministry <- scst_wide %>%
  filter(abs(st_excess) >= 0.3) %>%
  mutate(ministry_clean = fct_reorder(ministry_clean, st_excess),
         direction = if_else(st_excess > 0, "ST MPs ask more", "ST MPs ask less")) %>%
  ggplot(aes(x = st_excess, y = ministry_clean, fill = direction)) +
  geom_col(width = 0.75) +
  geom_vline(xintercept = 0, linewidth = 0.5, colour = "grey40") +
  scale_fill_manual(values = c("ST MPs ask more" = "#FF7F00",
                               "ST MPs ask less" = "#AAAAAA"),
                    name = NULL) +
  labs(
    title    = "ST reserved-seat MPs: ministry focus vs general-seat MPs",
    subtitle = "Percentage point difference in share of questions. Positive = ST MPs focus more here.",
    x = "pp difference (ST minus General)", y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", colour = NAVY),
        legend.position = "bottom")

p_scst_ministry <- p_sc_ministry / p_st_ministry +
  plot_annotation(
    title    = "Reserved-seat MPs target different ministries",
    subtitle = "SC and ST constituencies show distinct advocacy patterns vs general seats.",
    theme    = theme(plot.title    = element_text(face = "bold", colour = NAVY, size = 13),
                     plot.subtitle = element_text(size = 10, colour = "grey40"))
  )

ggsave(file.path(FIGDIR, "scst_ministry.png"),
       p_scst_ministry, width = 11, height = 11, dpi = 180)
cat("Saved: scst_ministry.png\n")
#}

# ============================================================
# SECTION 7: SC/ST — Party representation
# ============================================================
#{

scst_party <- starred %>%
  filter(!is.na(party_family)) %>%
  group_by(party_family) %>%
  summarise(
    total   = n(),
    sc_n    = sum(is_sc, na.rm=TRUE),
    st_n    = sum(is_st, na.rm=TRUE),
    scst_n  = sc_n + st_n,
    scst_pct = 100 * scst_n / total,
    .groups = "drop"
  ) %>%
  filter(total >= 30) %>%
  arrange(desc(scst_pct))

p_scst_party <- scst_party %>%
  pivot_longer(c(sc_n, st_n), names_to = "type", values_to = "n_q") %>%
  mutate(
    type         = recode(type, sc_n = "SC reserved", st_n = "ST reserved"),
    party_family = fct_reorder(party_family, scst_pct)
  ) %>%
  ggplot(aes(x = n_q, y = party_family, fill = type)) +
  geom_col(width = 0.75) +
  scale_fill_manual(values = c("SC reserved" = "#E31A1C",
                               "ST reserved" = "#FF7F00"),
                    name = NULL) +
  labs(
    title    = "SC/ST reserved-seat questions by party",
    subtitle = "Number of starred questions from MPs in SC or ST reserved constituencies.",
    x = "Number of starred questions", y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title      = element_text(face = "bold", colour = NAVY),
        legend.position = "bottom")

ggsave(file.path(FIGDIR, "scst_party.png"),
       p_scst_party, width = 10, height = 6, dpi = 180)
cat("Saved: scst_party.png\n")
#}

# ============================================================
# SECTION 8: Summary stats for QMD callouts
# ============================================================
#{
summary_stats <- list(
  female_q     = sum(starred$is_female, na.rm=TRUE),
  female_pct   = round(100 * mean(starred$is_female, na.rm=TRUE), 1),
  sc_q         = sum(starred$is_sc, na.rm=TRUE),
  st_q         = sum(starred$is_st, na.rm=TRUE),
  sc_pct       = round(100 * mean(starred$is_sc, na.rm=TRUE), 1),
  st_pct       = round(100 * mean(starred$is_st, na.rm=TRUE), 1),
  top_female_min = gender_ministry %>% filter(gap > 0) %>% slice_max(gap, n=1) %>% pull(ministry_clean),
  top_male_min   = gender_ministry %>% filter(gap < 0) %>% slice_min(gap, n=1) %>% pull(ministry_clean),
  top_sc_min   = scst_wide %>% slice_max(sc_excess, n=1) %>% pull(ministry_clean),
  top_st_min   = scst_wide %>% slice_max(st_excess, n=1) %>% pull(ministry_clean)
)

saveRDS(summary_stats, file.path(TABDIR, "gender_sc_summary.rds"))

cat("\n=== Summary ===\n")
cat(sprintf("Female questions: %d (%.1f%%)\n", summary_stats$female_q, summary_stats$female_pct))
cat(sprintf("SC questions: %d (%.1f%%)\n", summary_stats$sc_q, summary_stats$sc_pct))
cat(sprintf("ST questions: %d (%.1f%%)\n", summary_stats$st_q, summary_stats$st_pct))
cat(sprintf("Top female ministry: %s\n", summary_stats$top_female_min))
cat(sprintf("Top male ministry: %s\n", summary_stats$top_male_min))
cat(sprintf("Top SC ministry: %s\n", summary_stats$top_sc_min))
cat(sprintf("Top ST ministry: %s\n", summary_stats$top_st_min))
cat("\n=== F1 complete ===\n")
#}
