# =============================================================================
# O1_language_analysis.R
# Author: Piyush Zaware
# Updated: 2026-06-16
#
# Goal: Detect linguistic register of question titles in LS and RS.
#       All text is in Latin script (transliterated), so we measure:
#         (1) Hindi-origin vocabulary intensity (scheme names, policy terms)
#         (2) Regional language geographic markers (Tamil, Telugu, Bengali)
#         (3) English-dominant vs Hindi-dominant framing at party level
#
# Inputs:
#   tmp/train-0000[0-4]-of-00005.parquet   (LS)
#   tmp/rajyasabha_clean.parquet           (RS)
#   input/mp_name_crosswalk.csv
#   input/rs_name_crosswalk.csv
#   input/hindi_vocab_lexicon.csv
#
# Outputs:
#   output/tables/lang_question_level.csv
#   output/tables/lang_party_house.csv
#   output/figures/compare_hindi_intensity.png
#   output/figures/compare_regional_markers.png
#   output/figures/compare_lang_heatmap.png
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}
TABDIR <- file.path(OUTDIR, "tables")
FIGDIR <- file.path(OUTDIR, "figures")

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(tidytext)
  library(patchwork)
})

PARTY_COLORS <- c(
  BJP = "#FF6B35", INC = "#1565C0", Left = "#B71C1C", TMC = "#1A237E",
  SP  = "#E53935", BSP = "#6A1B9A", DMK = "#004D40", BJD = "#00695C",
  TDP = "#F9A825", TRS = "#00838F", JDU = "#2E7D32", AIADMK = "#AD1457",
  NCP = "#4527A0", AAP = "#00BCD4", RJD = "#FF8F00", Other = "#78909C"
)

cat("[O1] Loading data...\n")

# =============================================================================
# SECTION 1: Load matched questions for both houses
# =============================================================================
#{
cw_ls <- read_csv(file.path(INPDIR, "mp_name_crosswalk.csv"), show_col_types = FALSE)
cw_rs <- read_csv(file.path(INPDIR, "rs_name_crosswalk.csv"), show_col_types = FALSE)
lexicon <- read_csv(file.path(INPDIR, "hindi_vocab_lexicon.csv"), show_col_types = FALSE)

hindi_words   <- lexicon %>% filter(category == "scheme_hindi") %>% pull(word)
tamil_words   <- lexicon %>% filter(category == "tamil_marker") %>% pull(word)
telugu_words  <- lexicon %>% filter(category == "telugu_marker") %>% pull(word)
bengali_words <- lexicon %>% filter(category == "bengali_marker") %>% pull(word)

# Build regex patterns
re_hindi   <- str_c("\\b(", str_c(hindi_words,   collapse = "|"), ")\\b")
re_tamil   <- str_c("\\b(", str_c(tamil_words,   collapse = "|"), ")\\b")
re_telugu  <- str_c("\\b(", str_c(telugu_words,  collapse = "|"), ")\\b")
re_bengali <- str_c("\\b(", str_c(bengali_words, collapse = "|"), ")\\b")

# LS
ls_files <- list.files(TMPDIR, pattern = "train-0000[0-4]-of-00005\\.parquet",
                       full.names = TRUE)
ls_q <- bind_rows(lapply(ls_files, read_parquet)) %>%
  filter(type == "STARRED") %>%
  mutate(
    primary_raw = str_to_upper(str_squish(map_chr(members, ~.x[[1]]))),
    lok_no      = as.integer(lok_no)
  ) %>%
  left_join(cw_ls %>% select(raw_name, lok_no, party_family),
            by = c("primary_raw" = "raw_name", "lok_no")) %>%
  filter(!is.na(party_family), !is.na(subject))

# RS
rs_q <- read_parquet(file.path(TMPDIR, "rajyasabha_clean.parquet")) %>%
  mutate(
    qtype    = str_to_upper(str_trim(qtype)),
    year     = as.integer(str_sub(as.character(adate), 1, 4)),
    raw_name = str_squish(replace_na(as.character(name), "")),
    qtitle   = str_squish(replace_na(as.character(qtitle), ""))
  ) %>%
  filter(qtype == "STARRED", year >= 2014, nchar(raw_name) > 1, nchar(qtitle) > 3) %>%
  left_join(cw_rs %>% select(raw_name, party_family), by = "raw_name") %>%
  filter(!is.na(party_family))

cat(sprintf("  LS: %d | RS: %d\n", nrow(ls_q), nrow(rs_q)))
#}

# =============================================================================
# SECTION 2: Score each question for linguistic register
# =============================================================================
#{
cat("[O1] Scoring linguistic register...\n")

score_text <- function(text) {
  t <- str_to_lower(text)
  n_words <- lengths(str_split(str_trim(t), "\\s+"))
  hindi_hits   <- str_count(t, re_hindi)
  tamil_hits   <- str_count(t, re_tamil)
  telugu_hits  <- str_count(t, re_telugu)
  bengali_hits <- str_count(t, re_bengali)
  tibble(
    n_words      = n_words,
    hindi_hits   = hindi_hits,
    tamil_hits   = tamil_hits,
    telugu_hits  = telugu_hits,
    bengali_hits = bengali_hits,
    has_hindi    = hindi_hits   > 0,
    has_tamil    = tamil_hits   > 0,
    has_telugu   = telugu_hits  > 0,
    has_bengali  = bengali_hits > 0,
    has_regional = (tamil_hits + telugu_hits + bengali_hits) > 0,
    any_marker   = (hindi_hits + tamil_hits + telugu_hits + bengali_hits) > 0
  )
}

ls_scored <- ls_q %>%
  bind_cols(score_text(ls_q$subject)) %>%
  mutate(house = "Lok Sabha")

rs_scored <- rs_q %>%
  bind_cols(score_text(rs_q$qtitle)) %>%
  mutate(house = "Rajya Sabha")

# Save question-level scores
lang_q <- bind_rows(
  ls_scored %>% select(party_family, house, n_words, hindi_hits, tamil_hits,
                       telugu_hits, bengali_hits, has_hindi, has_tamil,
                       has_telugu, has_bengali, has_regional, any_marker),
  rs_scored %>% select(party_family, house, n_words, hindi_hits, tamil_hits,
                       telugu_hits, bengali_hits, has_hindi, has_tamil,
                       has_telugu, has_bengali, has_regional, any_marker)
)
write_csv(lang_q, file.path(TABDIR, "lang_question_level.csv"))
cat("  Saved lang_question_level.csv\n")
#}

# =============================================================================
# SECTION 3: Aggregate by party × house
# =============================================================================
#{
cat("[O1] Aggregating by party × house...\n")

key_parties <- c("BJP","INC","Left","TMC","SP","DMK","AIADMK","TDP","JDU","NCP","BSP","AAP","RJD","BJD")

lang_party <- lang_q %>%
  filter(party_family %in% key_parties) %>%
  group_by(party_family, house) %>%
  summarise(
    n_questions      = n(),
    pct_hindi        = 100 * mean(has_hindi,    na.rm = TRUE),
    pct_tamil        = 100 * mean(has_tamil,    na.rm = TRUE),
    pct_telugu       = 100 * mean(has_telugu,   na.rm = TRUE),
    pct_bengali      = 100 * mean(has_bengali,  na.rm = TRUE),
    pct_regional     = 100 * mean(has_regional, na.rm = TRUE),
    mean_hindi_hits  = mean(hindi_hits, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n_questions >= 20)

write_csv(lang_party, file.path(TABDIR, "lang_party_house.csv"))
cat("  Saved lang_party_house.csv\n")
print(lang_party %>% arrange(house, desc(pct_hindi)))
#}

# =============================================================================
# SECTION 4: Figure 1 - Hindi vocabulary intensity LS vs RS
# =============================================================================
#{
cat("[O1] Figure 1: Hindi vocabulary intensity...\n")

lang_wide <- lang_party %>%
  select(party_family, house, pct_hindi, n_questions) %>%
  pivot_wider(names_from = house, values_from = c(pct_hindi, n_questions)) %>%
  drop_na()

# Parties that appear in both houses
both_parties <- lang_wide$party_family

lang_filt <- lang_party %>%
  filter(party_family %in% both_parties)

party_order_hindi <- lang_filt %>%
  group_by(party_family) %>%
  summarise(avg = mean(pct_hindi)) %>%
  arrange(avg) %>%
  pull(party_family)

p1 <- ggplot(lang_filt,
             aes(x = factor(party_family, levels = party_order_hindi),
                 y = pct_hindi, colour = house, group = house)) +
  geom_point(aes(size = n_questions), alpha = 0.9) +
  geom_line(
    data = lang_wide %>%
      pivot_longer(starts_with("pct_hindi"),
                   names_to = "house", values_to = "pct_hindi") %>%
      mutate(house = if_else(str_detect(house, "Lok"), "Lok Sabha", "Rajya Sabha")),
    aes(x = party_family, y = pct_hindi, group = party_family),
    colour = "grey60", linewidth = 0.5, linetype = "dashed"
  ) +
  scale_colour_manual(
    values = c("Lok Sabha" = "#1A5276", "Rajya Sabha" = "#B7950B"), name = NULL
  ) +
  scale_size_continuous(range = c(3, 10), name = "Questions") +
  coord_flip() +
  labs(
    title    = "Hindi-origin vocabulary in question titles: LS vs RS",
    subtitle = "% of starred question titles containing at least one transliterated Hindi policy term\n(yojana, kisan, swachh, jal, vikas, mudra, etc.)",
    x = NULL, y = "% questions with Hindi-origin terms"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top", panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "compare_hindi_intensity.png"), p1,
       width = 8, height = 6, dpi = 150)
cat("  Saved compare_hindi_intensity.png\n")
#}

# =============================================================================
# SECTION 5: Figure 2 - Regional marker presence by party
# =============================================================================
#{
cat("[O1] Figure 2: regional geographic markers...\n")

regional_long <- lang_party %>%
  select(party_family, house, pct_tamil, pct_telugu, pct_bengali, n_questions) %>%
  pivot_longer(starts_with("pct_"), names_to = "region", values_to = "pct") %>%
  mutate(
    region = recode(region,
                    pct_tamil   = "Tamil markers",
                    pct_telugu  = "Telugu markers",
                    pct_bengali = "Bengali markers")
  ) %>%
  filter(pct > 0)

p2 <- ggplot(regional_long,
             aes(x = reorder(party_family, pct), y = pct, fill = house)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_manual(
    values = c("Lok Sabha" = "#1A5276", "Rajya Sabha" = "#B7950B"), name = NULL
  ) +
  facet_wrap(~region, scales = "free_x") +
  coord_flip() +
  labs(
    title    = "Regional geographic markers in question titles: LS vs RS",
    subtitle = "% of questions containing geographic terms specific to Tamil, Telugu, or Bengali regions",
    x = NULL, y = "% questions with regional marker"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top", panel.grid.major.y = element_blank(),
        strip.text = element_text(face = "bold"),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "compare_regional_markers.png"), p2,
       width = 10, height = 5, dpi = 150)
cat("  Saved compare_regional_markers.png\n")
#}

# =============================================================================
# SECTION 6: Figure 3 - Combined heatmap: linguistic register by party × house
# =============================================================================
#{
cat("[O1] Figure 3: linguistic register heatmap...\n")

heatmap_data <- lang_party %>%
  filter(party_family %in% both_parties) %>%
  select(party_family, house, pct_hindi, pct_regional) %>%
  pivot_longer(c(pct_hindi, pct_regional),
               names_to = "metric", values_to = "pct") %>%
  mutate(
    metric = recode(metric,
                    pct_hindi    = "Hindi policy terms",
                    pct_regional = "Regional geographic markers"),
    party_family = factor(party_family,
                          levels = party_order_hindi)
  )

p3 <- ggplot(heatmap_data,
             aes(x = house, y = party_family, fill = pct)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f%%", pct)),
            size = 3, colour = "white", fontface = "bold") +
  scale_fill_gradient(low = "#EEF2FF", high = "#1A237E",
                      name = "% questions") +
  facet_wrap(~metric, nrow = 1) +
  labs(
    title    = "Linguistic register of parliamentary questions",
    subtitle = "% of each party's questions in each house containing the given vocabulary type",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1),
        strip.text  = element_text(face = "bold", size = 11),
        plot.title  = element_text(face = "bold"),
        panel.grid  = element_blank())

ggsave(file.path(FIGDIR, "compare_lang_heatmap.png"), p3,
       width = 9, height = 6, dpi = 150)
cat("  Saved compare_lang_heatmap.png\n")
#}

cat("\n[O1] Language analysis complete.\n")
