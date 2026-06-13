# =============================================================================
# E1_sule_report.R — Supriya Sule Parliamentary Activity Report
# Author: Piyush Zaware
# Updated: 2026-06-13
#
# PURPOSE:
#   Generates all figures and tables for the Supriya Sule MP report.
#   Supriya Sule (NCP-SP, Baramati) has been an MP across all three Lok
#   Sabhas in this dataset (16th, 17th, 18th). Baramati is a sugar-belt
#   constituency in Pune district, Maharashtra.
#
# INPUTS:
#   tmp/train-*.parquet                        — all questions
#   input/mp_party_lookup.csv                  — party metadata
#   output/tables/bertopic_doc_assignments.csv — topic per question
#   output/tables/sentiment_doc.csv            — VADER scores
#
# OUTPUTS:
#   output/figures/sule_*.png   — all report figures
#   output/tables/sule_*.csv    — supporting tables
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

# Set paths (defined here so E1 can be sourced standalone or from sule_report.qmd)
if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

pkgs_e1 <- c("arrow", "tidyverse", "tidytext", "patchwork", "ggrepel",
              "igraph", "ggraph", "tidygraph", "stopwords")
to_inst <- pkgs_e1[!pkgs_e1 %in% rownames(installed.packages())]
if (length(to_inst) > 0) install.packages(to_inst)

suppressPackageStartupMessages({
  library(arrow)
  library(tidyverse)
  library(tidytext)
  library(patchwork)
  library(ggrepel)
  library(igraph)
  library(ggraph)
  library(tidygraph)
})

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")

SAFFRON <- "#FF6B35"
NAVY    <- "#0D1B2A"
GREEN   <- "#138808"
CREAM   <- "#F5E6D3"

# ============================================================
# SECTION 1: Load all starred questions + tag Sule
# ============================================================
#{

cat("Loading questions...\n")
parquet_files <- list.files(TMPDIR, pattern = "train-.*\\.parquet$",
                             full.names = TRUE)

raw <- purrr::map_dfr(parquet_files, function(f) {
  read_parquet(f, col_select = c("id","lok_no","session_no","type",
                                  "ministry","members","question_text"))
})

starred <- raw %>% filter(type == "STARRED")
cat("  Total starred:", nrow(starred), "\n")

# Extract member list
starred <- starred %>%
  mutate(member_list = purrr::map(members, function(x) {
    tryCatch(str_squish(as.character(x)), error = function(e) character(0))
  }))

# Tag Sule questions
is_sule <- function(mlist) {
  any(str_detect(str_to_upper(mlist), "SUPRIYA") &
      (str_detect(str_to_upper(mlist), "SULE") |
       str_detect(str_to_upper(mlist), "SADANAND")))
}

starred <- starred %>%
  mutate(
    has_sule  = purrr::map_lgl(member_list, is_sule),
    is_primary = purrr::map_lgl(member_list, function(ml) {
      length(ml) > 0 &&
        str_detect(str_to_upper(ml[[1]]), "SUPRIYA") &&
        (str_detect(str_to_upper(ml[[1]]), "SULE") |
         str_detect(str_to_upper(ml[[1]]), "SADANAND"))
    })
  )

sule <- starred %>% filter(has_sule)
cat("  Supriya Sule starred questions:", nrow(sule), "\n")
cat("  Primary questioner:", sum(sule$is_primary), "\n")
cat("  Co-signatory:", sum(!sule$is_primary), "\n\n")

# Ministry cleaning (same map as C1)
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
  "COAL"                            = "Coal",
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
  "INFORMATION AND BROADCASTING"    = "I&B"
)

sule <- sule %>%
  mutate(ministry_clean = recode(str_to_upper(str_trim(ministry)),
                                  !!!ministry_recode,
                                  .default = str_to_title(str_trim(ministry))))

starred_others <- starred %>%
  filter(!has_sule) %>%
  mutate(ministry_clean = recode(str_to_upper(str_trim(ministry)),
                                  !!!ministry_recode,
                                  .default = str_to_title(str_trim(ministry))))
#}

# ============================================================
# SECTION 2: Ministry profile — Sule vs national average
# ============================================================
#{

sule_min <- sule %>%
  count(ministry_clean, name = "sule_n") %>%
  mutate(sule_share = sule_n / sum(sule_n))

all_min <- starred %>%
  mutate(ministry_clean = recode(str_to_upper(str_trim(ministry)),
                                  !!!ministry_recode,
                                  .default = str_to_title(str_trim(ministry)))) %>%
  count(ministry_clean, name = "all_n") %>%
  mutate(all_share = all_n / sum(all_n))

min_compare <- sule_min %>%
  left_join(all_min, by = "ministry_clean") %>%
  mutate(
    excess     = sule_share - all_share,
    excess_pct = 100 * excess
  ) %>%
  filter(sule_n >= 1) %>%
  arrange(desc(sule_share))

write_csv(min_compare, file.path(TABDIR, "sule_ministry.csv"))

# Bar chart: her share vs national share
top_min <- min_compare %>% slice_max(sule_n, n = 15)

p_ministry <- top_min %>%
  pivot_longer(c(sule_share, all_share),
               names_to = "who", values_to = "share") %>%
  mutate(
    who   = recode(who, sule_share = "Supriya Sule",
                        all_share  = "All MPs (national avg)"),
    ministry_clean = fct_reorder(ministry_clean, share, .fun = max)
  ) %>%
  ggplot(aes(x = ministry_clean, y = share * 100, fill = who)) +
  geom_col(position = "dodge", width = 0.65) +
  coord_flip() +
  scale_fill_manual(values = c("Supriya Sule" = SAFFRON,
                               "All MPs (national avg)" = "#AAAAAA"),
                    name = NULL) +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  labs(
    title    = "Ministry focus: Supriya Sule vs national average",
    subtitle = "Share of starred questions directed at each ministry.",
    x = NULL, y = "% of starred questions",
    caption  = paste0("Supriya Sule: ", nrow(sule), " starred questions across 16th–18th Lok Sabha.")
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title      = element_text(face = "bold", colour = NAVY),
    axis.text.y     = element_text(size = 10)
  )

ggsave(file.path(FIGDIR, "sule_ministry.png"),
       p_ministry, width = 10, height = 7, dpi = 180)
cat("Saved: sule_ministry.png\n")
#}

# ============================================================
# SECTION 3: Temporal activity — questions per session
# ============================================================
#{

temporal <- sule %>%
  count(lok_no, session_no, is_primary) %>%
  mutate(
    lok_label     = paste0(lok_no, "th LS"),
    session_label = paste0("S", session_no),
    role          = if_else(is_primary, "Primary", "Co-signatory")
  )

p_temporal <- ggplot(temporal,
                     aes(x = session_no, y = n, fill = role)) +
  geom_col(width = 0.75) +
  facet_wrap(~ lok_label, scales = "free_x", nrow = 1) +
  scale_fill_manual(values = c(Primary = SAFFRON, `Co-signatory` = "#AAAAAA"),
                    name = NULL) +
  labs(
    title    = "Supriya Sule: starred questions by session",
    subtitle = "Split by whether she was the primary questioner or a co-signatory.",
    x = "Session", y = "Number of starred questions"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title      = element_text(face = "bold", colour = NAVY),
    strip.text      = element_text(face = "bold")
  )

ggsave(file.path(FIGDIR, "sule_temporal.png"),
       p_temporal, width = 10, height = 5, dpi = 180)
cat("Saved: sule_temporal.png\n")
#}

# ============================================================
# SECTION 4: Distinctive vocabulary (TF-IDF vs all other MPs)
# ============================================================
#{

# Combine all Sule question texts into one document; all others into another
clean_text <- function(t) {
  if (is.na(t)) return("")
  t <- str_replace_all(t, "##.*?\\n", " ")
  t <- str_replace_all(t, "\\([a-z]\\)", " ")
  t <- str_squish(t)
  t
}

sule_corpus <- sule %>%
  mutate(text_c = map_chr(question_text, clean_text)) %>%
  filter(nchar(text_c) > 30)

# TF-IDF: Sule vs rest-of-parliament
tfidf_corpus <- bind_rows(
  sule_corpus %>% mutate(doc = "Supriya Sule") %>% select(doc, text_c),
  starred_others %>%
    mutate(text_c = map_chr(question_text, clean_text),
           doc    = "Other MPs") %>%
    filter(nchar(text_c) > 30) %>%
    select(doc, text_c)
)

custom_stop <- c("will", "minister", "whether", "government", "please",
                  "state", "details", "thereof", "taken", "steps",
                  "also", "further", "said", "country", "india",
                  "hon", "aware", "regard", "provide", "information",
                  "thereon", "thereunder", "proposed", "considered",
                  "members", "question", "starred", "unstarred",
                  "lok", "sabha", "rajya", "session", "parliament",
                  stopwords::stopwords("en"))

word_tfidf <- tfidf_corpus %>%
  unnest_tokens(word, text_c) %>%
  filter(!word %in% custom_stop,
         str_detect(word, "^[a-z]+$"),
         nchar(word) >= 4) %>%
  count(doc, word) %>%
  bind_tf_idf(word, doc, n) %>%
  filter(doc == "Supriya Sule") %>%
  arrange(desc(tf_idf))

write_csv(word_tfidf, file.path(TABDIR, "sule_tfidf.csv"))

p_tfidf <- word_tfidf %>%
  slice_max(tf_idf, n = 25) %>%
  mutate(word = fct_reorder(word, tf_idf)) %>%
  ggplot(aes(x = word, y = tf_idf, fill = tf_idf)) +
  geom_col(width = 0.75, show.legend = FALSE) +
  coord_flip() +
  scale_fill_gradient(low = CREAM, high = SAFFRON) +
  labs(
    title    = "Words most distinctive to Supriya Sule's questions",
    subtitle = "TF-IDF score: how much more she uses each word compared to all other MPs combined.",
    x = NULL, y = "TF-IDF score"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title  = element_text(face = "bold", colour = NAVY),
    axis.text.y = element_text(size = 10)
  )

ggsave(file.path(FIGDIR, "sule_tfidf.png"),
       p_tfidf, width = 9, height = 8, dpi = 180)
cat("Saved: sule_tfidf.png\n")
#}

# ============================================================
# SECTION 5: BERTopic topic distribution
# ============================================================
#{

bert_path <- file.path(TABDIR, "bertopic_doc_assignments.csv")
word_path <- file.path(TABDIR, "bertopic_topic_words.csv")

if (file.exists(bert_path) && file.exists(word_path)) {
  bert_docs  <- read_csv(bert_path,  show_col_types = FALSE)
  bert_words <- read_csv(word_path,  show_col_types = FALSE)

  topic_labels <- bert_words %>%
    group_by(topic_id) %>%
    slice_max(score, n = 3) %>%
    summarise(label = paste(word, collapse = " / "), .groups = "drop")

  sule_topics <- sule %>%
    left_join(bert_docs %>% select(id, topic_id), by = "id") %>%
    filter(!is.na(topic_id), topic_id != -1) %>%
    count(topic_id) %>%
    left_join(topic_labels, by = "topic_id") %>%
    filter(!is.na(label)) %>%
    mutate(share = n / sum(n)) %>%
    arrange(desc(n))

  # All MPs topic distribution for comparison
  all_topics <- bert_docs %>%
    filter(topic_id != -1) %>%
    count(topic_id) %>%
    mutate(all_share = n / sum(n)) %>%
    rename(all_n = n)

  sule_topics <- sule_topics %>%
    left_join(all_topics, by = "topic_id") %>%
    mutate(excess = share - all_share)

  write_csv(sule_topics, file.path(TABDIR, "sule_topics.csv"))

  p_topics <- sule_topics %>%
    slice_max(n, n = 12) %>%
    mutate(label = str_trunc(label, 30),
           label = fct_reorder(label, n)) %>%
    ggplot(aes(x = label, y = n, fill = excess * 100)) +
    geom_col(width = 0.75) +
    coord_flip() +
    scale_fill_gradient2(
      low = "#1F78B4", mid = "grey90", high = SAFFRON,
      midpoint = 0, name = "Excess\nvs avg (%)"
    ) +
    labs(
      title    = "Supriya Sule: BERTopic topic distribution",
      subtitle = "Number of her starred questions in each topic cluster.\nFill = how much more she focuses on this topic vs the parliament-wide average.",
      x = NULL, y = "Number of questions"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title  = element_text(face = "bold", colour = NAVY),
      axis.text.y = element_text(size = 9)
    )

  ggsave(file.path(FIGDIR, "sule_topics.png"),
         p_topics, width = 10, height = 6, dpi = 180)
  cat("Saved: sule_topics.png\n")
}
#}

# ============================================================
# SECTION 6: Sentiment vs NCP peers
# ============================================================
#{

sent_path <- file.path(TABDIR, "sentiment_doc.csv")
if (file.exists(sent_path)) {
  sent_all <- read_csv(sent_path, show_col_types = FALSE)

  # Sule sentiment
  sule_sent <- sent_all %>%
    filter(id %in% sule$id)

  # INC/NCP peers for comparison
  peer_sent <- sent_all %>%
    filter(party_family == "INC", !id %in% sule$id)

  cat("Sule mean compound:", round(mean(sule_sent$sent_compound, na.rm=TRUE), 3), "\n")
  cat("INC peers mean compound:", round(mean(peer_sent$sent_compound, na.rm=TRUE), 3), "\n")
  cat("Sule adversarial rate:",
      round(100*mean(sule_sent$sent_compound <= -0.05, na.rm=TRUE), 1), "%\n")

  # Distribution comparison
  compare_sent <- bind_rows(
    sule_sent %>% mutate(who = "Supriya Sule"),
    peer_sent  %>% sample_n(min(200, nrow(peer_sent))) %>%
                   mutate(who = "INC/NCP peers")
  )

  p_sent <- ggplot(compare_sent,
                   aes(x = sent_compound, fill = who, colour = who)) +
    geom_density(alpha = 0.35, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    scale_fill_manual(values   = c("Supriya Sule" = SAFFRON,
                                   "INC/NCP peers" = NAVY),
                      name = NULL) +
    scale_colour_manual(values = c("Supriya Sule" = SAFFRON,
                                   "INC/NCP peers" = NAVY),
                        name = NULL) +
    labs(
      title    = "Question tone: Supriya Sule vs INC/NCP peers",
      subtitle = "VADER compound score distribution. Left of 0 = more adversarial framing.",
      x = "VADER compound score", y = "Density"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "bottom",
      plot.title      = element_text(face = "bold", colour = NAVY)
    )

  ggsave(file.path(FIGDIR, "sule_sentiment.png"),
         p_sent, width = 9, height = 5, dpi = 180)
  cat("Saved: sule_sentiment.png\n")
}
#}

# ============================================================
# SECTION 7: Co-signatory network
# ============================================================
#{

# Extract all co-signatories (not Sule herself)
mp_lookup <- read_csv(file.path(INPDIR, "mp_party_lookup.csv"),
                      show_col_types = FALSE) %>%
  mutate(mp_key = str_to_upper(str_squish(mp_name))) %>%
  arrange(desc(lok_no)) %>%
  distinct(mp_key, .keep_all = TRUE)

cosign_rows <- sule %>%
  purrr::pmap_dfr(function(id, lok_no, session_no, member_list, ...) {
    others <- member_list[!str_detect(str_to_upper(member_list),
                                      "SUPRIYA|SADANAND|SULE")]
    if (length(others) == 0) return(NULL)
    tibble(cosignatory = str_to_upper(str_squish(others)),
           lok_no = lok_no, question_id = id)
  })

cosign_count <- cosign_rows %>%
  count(cosignatory, name = "n_questions") %>%
  left_join(mp_lookup %>% select(mp_key, party_family, constituency),
            by = c("cosignatory" = "mp_key")) %>%
  arrange(desc(n_questions))

write_csv(cosign_count, file.path(TABDIR, "sule_cosignatories.csv"))

PARTY_COLORS <- c(
  BJP="#FF6B35", INC="#1F78B4", Left="#E31A1C", BSP="#6A3D9A",
  AAP="#33A02C", TMC="#B15928", SP="#FF7F00", JDU="#A6CEE3",
  DMK="#B2DF8A", TDP="#FDBF6F", `Shiv Sena`="#FB9A99",
  RJD="#CAB2D6", NCP="#999999", Regional="#CCCCCC"
)

p_cosign <- cosign_count %>%
  filter(n_questions >= 1) %>%
  slice_max(n_questions, n = 15) %>%
  mutate(
    name_clean = str_to_title(cosignatory),
    fill_col   = coalesce(PARTY_COLORS[party_family], "#AAAAAA")
  ) %>%
  ggplot(aes(x = reorder(name_clean, n_questions),
             y = n_questions, fill = party_family)) +
  geom_col(width = 0.75) +
  geom_text(aes(label = paste0(party_family, " · ",
                               str_to_title(str_trunc(constituency, 20, ellipsis="")))),
            hjust = -0.05, size = 2.8, colour = "grey30") +
  coord_flip() +
  scale_fill_manual(values = PARTY_COLORS, na.value = "#AAAAAA",
                    guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.45))) +
  labs(
    title    = "Supriya Sule's most frequent co-signatories",
    subtitle = "MPs who co-signed the most starred questions with her.",
    x = NULL, y = "Questions co-signed",
    caption  = "All co-signatories are NCP/INC Maharashtra MPs, confirming tight regional coordination."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title  = element_text(face = "bold", colour = NAVY),
    axis.text.y = element_text(size = 9.5)
  )

ggsave(file.path(FIGDIR, "sule_cosignatories.png"),
       p_cosign, width = 11, height = 7, dpi = 180)
cat("Saved: sule_cosignatories.png\n")
#}

# ============================================================
# SECTION 8: Ministry excess targeting vs national parliament
# ============================================================
#{

# Pearson residual for each Sule-ministry pair
national_total  <- nrow(starred)
national_by_min <- starred %>%
  mutate(ministry_clean = recode(str_to_upper(str_trim(ministry)),
                                  !!!ministry_recode,
                                  .default = str_to_title(str_trim(ministry)))) %>%
  count(ministry_clean, name = "nat_n")

sule_excess <- sule %>%
  count(ministry_clean, name = "sule_n") %>%
  left_join(national_by_min, by = "ministry_clean") %>%
  mutate(
    expected      = nrow(sule) * (nat_n / national_total),
    pearson_resid = (sule_n - expected) / sqrt(pmax(expected, 0.5))
  ) %>%
  filter(sule_n >= 1) %>%
  arrange(desc(pearson_resid))

p_excess <- ggplot(sule_excess,
                   aes(x = reorder(ministry_clean, pearson_resid),
                       y = pearson_resid,
                       fill = pearson_resid > 0)) +
  geom_col(width = 0.75) +
  geom_hline(yintercept = 0, linewidth = 0.5, colour = "grey40") +
  coord_flip() +
  scale_fill_manual(values = c(`TRUE` = SAFFRON, `FALSE` = "#4A90D9"),
                    guide = "none") +
  labs(
    title    = "Where Supriya Sule over- and under-questions relative to parliament",
    subtitle = "Pearson residual: (observed - expected) / sqrt(expected).\nPositive = she questions this ministry more than her overall size would predict.",
    x = NULL, y = "Pearson residual"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title  = element_text(face = "bold", colour = NAVY),
    axis.text.y = element_text(size = 9.5)
  )

ggsave(file.path(FIGDIR, "sule_excess.png"),
       p_excess, width = 10, height = 7, dpi = 180)
cat("Saved: sule_excess.png\n")
#}

# ============================================================
# SECTION 9: Sample questions — most typical and most distinctive
# ============================================================
#{

# Most typical: highest BERTopic probability for her top topic
# Most distinctive: highest TF-IDF score words in question text
if (exists("sule_topics") && nrow(sule_topics) > 0) {
  top_topic <- sule_topics$topic_id[1]
  top_topic_label <- sule_topics$label[1]

  typical_q <- sule %>%
    left_join(bert_docs %>% select(id, topic_id), by = "id") %>%
    filter(topic_id == top_topic) %>%
    mutate(text_c = map_chr(question_text, clean_text)) %>%
    slice_head(n = 3) %>%
    select(id, lok_no, session_no, ministry_clean, text_c)

  write_csv(typical_q, file.path(TABDIR, "sule_sample_questions.csv"))
  cat("Saved: sule_sample_questions.csv\n")
  cat(sprintf("Top topic: %s (topic %d)\n", top_topic_label, top_topic))
}
#}

# ============================================================
# SECTION 10: Topic evolution across Lok Sabhas
# ============================================================
#{

ls_labels <- c("16" = "16th LS (2014-19)", "17" = "17th LS (2019-24)", "18" = "18th LS (2024-)")

# Part A: TF-IDF distinctive words per Lok Sabha period
sule_by_ls <- sule %>%
  mutate(text_c   = map_chr(question_text, clean_text),
         lok_label = ls_labels[as.character(lok_no)]) %>%
  filter(nchar(text_c) > 30)

tfidf_by_ls <- sule_by_ls %>%
  unnest_tokens(word, text_c) %>%
  filter(!word %in% custom_stop,
         str_detect(word, "^[a-z]+$"),
         nchar(word) >= 4) %>%
  count(lok_label, word) %>%
  bind_tf_idf(word, lok_label, n) %>%
  group_by(lok_label) %>%
  slice_max(tf_idf, n = 10, with_ties = FALSE) %>%
  ungroup()

p_topic_evolution <- tfidf_by_ls %>%
  mutate(word = reorder_within(word, tf_idf, lok_label),
         lok_label = factor(lok_label, levels = ls_labels)) %>%
  ggplot(aes(x = word, y = tf_idf,
             fill = factor(lok_label, levels = ls_labels))) +
  geom_col(show.legend = FALSE, width = 0.75) +
  facet_wrap(~ lok_label, scales = "free_y", ncol = 1) +
  scale_x_reordered() +
  scale_fill_manual(values = c(
    "16th LS (2014-19)" = NAVY,
    "17th LS (2019-24)" = SAFFRON,
    "18th LS (2024-)"   = GREEN
  )) +
  coord_flip() +
  labs(
    title    = "How Supriya Sule's vocabulary shifted across Lok Sabhas",
    subtitle = "Top TF-IDF words within each period. Shows which issues she emphasised in each term.",
    x = NULL, y = "TF-IDF score"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title  = element_text(face = "bold", colour = NAVY),
    strip.text  = element_text(face = "bold", size = 10),
    axis.text.y = element_text(size = 9)
  )

ggsave(file.path(FIGDIR, "sule_topic_evolution.png"),
       p_topic_evolution, width = 9, height = 11, dpi = 180)
cat("Saved: sule_topic_evolution.png\n")

# Part B: Ministry share slope chart across Lok Sabhas
top8_min <- sule %>%
  count(ministry_clean) %>%
  slice_max(n, n = 8) %>%
  pull(ministry_clean)

min_by_ls <- sule %>%
  filter(ministry_clean %in% top8_min) %>%
  count(lok_no, ministry_clean) %>%
  group_by(lok_no) %>%
  mutate(share = 100 * n / sum(n),
         lok_label = ls_labels[as.character(lok_no)]) %>%
  ungroup() %>%
  mutate(lok_label = factor(lok_label, levels = ls_labels))

p_ministry_trend <- ggplot(min_by_ls,
                           aes(x = lok_label, y = share,
                               group = ministry_clean,
                               colour = ministry_clean)) +
  geom_line(linewidth = 1, alpha = 0.85) +
  geom_point(size = 3) +
  geom_text_repel(
    data = min_by_ls %>% filter(lok_no == max(lok_no)),
    aes(label = ministry_clean),
    hjust      = -0.1, size = 3.2, segment.alpha = 0.4
  ) +
  scale_colour_brewer(palette = "Set2", guide = "none") +
  scale_x_discrete(expand = expansion(mult = c(0.05, 0.35))) +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  labs(
    title    = "Ministry focus over time",
    subtitle = "Share of her starred questions going to each ministry, by Lok Sabha.",
    x = NULL, y = "% of questions"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title  = element_text(face = "bold", colour = NAVY),
    axis.text.x = element_text(size = 10)
  )

ggsave(file.path(FIGDIR, "sule_ministry_trend.png"),
       p_ministry_trend, width = 10, height = 6, dpi = 180)
cat("Saved: sule_ministry_trend.png\n")
#}

# ============================================================
# SECTION 11: Ministry persistence (repeat targeting)
# ============================================================
#{

# Session label: chronological order across all three LS
session_df <- sule %>%
  distinct(lok_no, session_no) %>%
  arrange(lok_no, session_no) %>%
  mutate(session_label = paste0(lok_no, "·S", session_no))

session_order <- session_df$session_label

persistence_heat <- sule %>%
  mutate(session_label = paste0(lok_no, "·S", session_no)) %>%
  count(ministry_clean, session_label) %>%
  group_by(ministry_clean) %>%
  mutate(
    total_q    = sum(n),
    n_sessions = n_distinct(session_label)
  ) %>%
  ungroup() %>%
  filter(total_q >= 2)

# Build full grid so empty sessions show as grey
all_combos <- expand_grid(
  ministry_clean = unique(persistence_heat$ministry_clean),
  session_label  = session_order
)

heat_full <- all_combos %>%
  left_join(persistence_heat %>% select(ministry_clean, session_label, n),
            by = c("ministry_clean", "session_label")) %>%
  mutate(n = replace_na(n, 0))

# Order ministries by total questions
min_order <- persistence_heat %>%
  distinct(ministry_clean, total_q) %>%
  arrange(desc(total_q)) %>%
  pull(ministry_clean)

# Add Lok Sabha dividers
ls_breaks <- session_df %>%
  group_by(lok_no) %>%
  slice_tail(n = 1) %>%
  pull(session_label)

p_persistence <- heat_full %>%
  mutate(
    session_label  = factor(session_label, levels = session_order),
    ministry_clean = factor(ministry_clean, levels = rev(min_order)),
    label_val      = if_else(n > 0, as.character(n), "")
  ) %>%
  ggplot(aes(x = session_label, y = ministry_clean, fill = n)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = label_val), size = 3, colour = "white", fontface = "bold") +
  scale_fill_gradient(low = "#EAD5C0", high = NAVY,
                      name = "Questions", limits = c(0, NA)) +
  geom_vline(
    xintercept = which(session_order %in% ls_breaks[1:2]) + 0.5,
    colour = SAFFRON, linewidth = 1, linetype = "dashed"
  ) +
  labs(
    title    = "Ministry targeting across every session",
    subtitle = "Each cell = number of starred questions. Dashed lines separate Lok Sabhas.\nReturning to a ministry across sessions signals a sustained accountability campaign.",
    x = "Session (16th LS → 17th LS → 18th LS)", y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 50, hjust = 1, size = 8),
    axis.text.y = element_text(size = 9.5),
    plot.title  = element_text(face = "bold", colour = NAVY),
    legend.position = "right"
  )

ggsave(file.path(FIGDIR, "sule_persistence.png"),
       p_persistence, width = 13, height = 6, dpi = 180)
cat("Saved: sule_persistence.png\n")

# Persistence summary table
persist_summary <- persistence_heat %>%
  distinct(ministry_clean, total_q, n_sessions) %>%
  arrange(desc(n_sessions), desc(total_q)) %>%
  mutate(persistence_score = round(n_sessions / n_distinct(sule$lok_no, sule$session_no), 2))

write_csv(persist_summary, file.path(TABDIR, "sule_persistence.csv"))
#}

# ============================================================
# SECTION 12: Peer benchmarking within Maharashtra MPs
# ============================================================
#{

# Normalize name: keep first + last token only.
# "SUPRIYA SADANAND SULE" → "SUPRIYA SULE" to match the lookup.
norm_fl <- function(x) {
  parts <- str_split(str_squish(x), "\\s+")[[1]]
  if (length(parts) <= 2) return(x)
  paste(parts[1], parts[length(parts)])
}

MAHA_PATTERNS <- paste(c(
  "MUMBAI", "PUNE", "BARAMATI", "NASHIK", "NAGPUR", "THANE",
  "KALYAN", "AURANGABAD", "KOLHAPUR", "SATARA", "SOLAPUR",
  "NANDED", "LATUR", "OSMANABAD", "PARBHANI", "HINGOLI",
  "JALGAON", "RAVER", "DHULE", "NANDURBAR", "AMRAVATI",
  "WARDHA", "CHANDRAPUR", "GADCHIROLI", "RAMTEK", "BHANDARA",
  "YAVATMAL", "AKOLA", "BULDHANA", "SHIRDI", "AHMADNAGAR",
  "MAVAL", "SHIRUR", "RAIGAD", "RATNAGIRI", "SINDHUDURG",
  "HATKANANGALE", "BHIWANDI", "DINDORI", "PALGHAR",
  "VIKRAMGAD", "DOMBIVALI"
), collapse = "|")

# Maharashtra lookup: canonical key = lookup mp_key; norm_key for fuzzy matching
maha_lookup <- mp_lookup %>%
  filter(str_detect(str_to_upper(coalesce(constituency, "")), MAHA_PATTERNS)) %>%
  mutate(norm_key = vapply(mp_key, norm_fl, character(1))) %>%
  arrange(desc(lok_no)) %>%
  distinct(norm_key, .keep_all = TRUE)   # one row per unique normalized name

cat(sprintf("  Maharashtra MPs in lookup: %d\n", nrow(maha_lookup)))

# Build norm_key → canonical lookup key map
norm_to_canonical <- setNames(maha_lookup$mp_key, maha_lookup$norm_key)
maha_canonical_keys <- maha_lookup$mp_key

# Extract primary MP for ALL starred questions, then normalize and join
starred_with_primary <- starred %>%
  mutate(
    primary_raw  = str_to_upper(str_squish(
      map_chr(member_list, function(ml) {
        if (length(ml) == 0) return(NA_character_)
        ml[1]
      })
    )),
    primary_norm = vapply(primary_raw, norm_fl, character(1))
  ) %>%
  filter(!is.na(primary_raw)) %>%
  # Map normalized parquet name → canonical lookup key
  mutate(
    primary_mp = coalesce(
      norm_to_canonical[primary_norm],   # matched via first+last name
      primary_raw                         # fallback: keep raw
    )
  )

# Maharashtra MPs: those whose canonical key is in the lookup
maha_starred <- starred_with_primary %>%
  filter(primary_mp %in% maha_canonical_keys) %>%
  mutate(ministry_clean = recode(str_to_upper(str_trim(ministry)),
                                  !!!ministry_recode,
                                  .default = str_to_title(str_trim(ministry))))

cat(sprintf("  Maharashtra starred questions found: %d\n", nrow(maha_starred)))

# Verify Sule is present
sule_check <- maha_starred %>% filter(str_detect(primary_mp, "SUPRIYA"))
cat(sprintf("  Supriya Sule questions in benchmark: %d\n", nrow(sule_check)))

# ── Metric 1: Total starred questions (primary only) ──
q_count <- maha_starred %>%
  count(primary_mp, name = "n_questions") %>%
  arrange(desc(n_questions))

# ── Metric 2: Ministry HHI (lower = more diverse) ──
min_hhi <- maha_starred %>%
  count(primary_mp, ministry_clean) %>%
  group_by(primary_mp) %>%
  mutate(share = n / sum(n)) %>%
  summarise(hhi = round(sum(share^2), 3), .groups = "drop")

# ── Metric 3: Co-signatory reach ──
# Build norm-key set for all Maharashtra MPs for member_list matching
maha_norm_set <- maha_lookup$norm_key

cosign_reach <- starred %>%
  filter(map_lgl(member_list, function(ml) {
    norms <- vapply(str_to_upper(str_squish(ml)), norm_fl, character(1))
    any(norms %in% maha_norm_set)
  })) %>%
  purrr::pmap_dfr(function(member_list, ...) {
    raw_mps   <- str_to_upper(str_squish(member_list))
    norm_mps  <- vapply(raw_mps, norm_fl, character(1))
    canon_mps <- coalesce(norm_to_canonical[norm_mps], raw_mps)
    maha_in   <- canon_mps[canon_mps %in% maha_canonical_keys]
    others    <- setdiff(raw_mps, raw_mps[canon_mps %in% maha_canonical_keys])
    if (length(maha_in) == 0) return(NULL)
    purrr::map_dfr(maha_in, ~ tibble(focal_mp = .x, partner = others))
  }) %>%
  group_by(focal_mp) %>%
  summarise(cosign_reach = n_distinct(partner[partner != ""]), .groups = "drop")

# ── Metric 4: Adversarial rate ──
if (file.exists(file.path(TABDIR, "sentiment_doc.csv"))) {
  sent_doc <- read_csv(file.path(TABDIR, "sentiment_doc.csv"),
                       show_col_types = FALSE)
  sent_with_mp <- starred_with_primary %>%
    select(id, primary_mp) %>%
    inner_join(sent_doc %>% select(id, sent_compound), by = "id") %>%
    filter(primary_mp %in% maha_canonical_keys) %>%
    group_by(primary_mp) %>%
    summarise(
      adv_rate   = round(100 * mean(sent_compound <= -0.05, na.rm = TRUE), 1),
      mean_score = round(mean(sent_compound, na.rm = TRUE), 3),
      .groups    = "drop"
    )
} else {
  sent_with_mp <- tibble(primary_mp = character(), adv_rate = numeric(),
                         mean_score = numeric())
}

# ── Combine & compute display names ──
bench <- q_count %>%
  left_join(min_hhi,      by = "primary_mp") %>%
  left_join(cosign_reach, by = c("primary_mp" = "focal_mp")) %>%
  left_join(sent_with_mp, by = "primary_mp") %>%
  left_join(maha_lookup %>% select(mp_key, constituency),
            by = c("primary_mp" = "mp_key")) %>%
  mutate(
    is_sule_flag = str_detect(primary_mp, "SUPRIYA"),
    name_clean   = if_else(is_sule_flag, "Supriya Sule",
                            str_to_title(primary_mp)),
    cosign_reach = replace_na(cosign_reach, 0),
    hhi          = replace_na(hhi, 1)
  ) %>%
  filter(n_questions >= 3)

write_csv(bench, file.path(TABDIR, "sule_peer_benchmark.csv"))
cat(sprintf("  Maharashtra MPs in benchmark (>= 3 Qs): %d\n", nrow(bench)))
cat(sprintf("  Sule in benchmark: %s\n",
            if (any(bench$is_sule_flag)) "YES" else "NO — check name matching"))

# Pre-compute ranks for QMD callout (written to a small csv)
bench_ranks <- bench %>%
  mutate(
    rank_q   = rank(-n_questions, ties.method = "min"),
    rank_hhi = rank(hhi, ties.method = "min"),
    rank_cs  = rank(-cosign_reach, ties.method = "min")
  )
sule_ranks <- bench_ranks %>% filter(is_sule_flag)
write_csv(bench_ranks, file.path(TABDIR, "sule_bench_ranks.csv"))

# ── Plot ──
build_rank_plot <- function(data, x_var, x_label,
                             sule_colour = SAFFRON, peer_colour = "#BBBBBB",
                             reverse = FALSE) {
  data <- data %>%
    filter(!is.na(.data[[x_var]])) %>%
    arrange(if (reverse) desc(.data[[x_var]]) else .data[[x_var]]) %>%
    mutate(name_disp = if_else(is_sule_flag,
                                paste0("★ ", name_clean), name_clean))
  ggplot(data, aes(x = .data[[x_var]],
                   y = reorder(name_disp,
                               if (reverse) -.data[[x_var]] else .data[[x_var]]),
                   fill = is_sule_flag)) +
    geom_col(width = 0.72) +
    scale_fill_manual(values = c(`TRUE` = sule_colour, `FALSE` = peer_colour),
                      guide = "none") +
    labs(x = x_label, y = NULL) +
    theme_minimal(base_size = 9) +
    theme(axis.text.y = element_text(size = 8,
                                      face = if_else(data$is_sule_flag[
                                        order(if (reverse) -data[[x_var]] else data[[x_var]])
                                      ], "bold", "plain")),
          axis.title.x = element_text(size = 9, colour = "grey40"))
}

p_bench_q   <- build_rank_plot(bench, "n_questions",  "Starred questions (primary)", reverse = TRUE)
p_bench_hhi <- build_rank_plot(bench, "hhi", "Ministry HHI\n(lower = more diverse)")
p_bench_cs  <- build_rank_plot(bench, "cosign_reach", "Co-signatory reach", reverse = TRUE)

p_bench <- if (nrow(sent_with_mp) > 0 && any(!is.na(bench$adv_rate))) {
  p_bench_adv <- build_rank_plot(
    bench %>% filter(!is.na(adv_rate)),
    "adv_rate", "Adversarial rate (%)", reverse = TRUE
  )
  (p_bench_q | p_bench_hhi) / (p_bench_cs | p_bench_adv) +
    plot_annotation(
      title    = "Supriya Sule vs Maharashtra MP peers",
      subtitle = "Ranked on four metrics. Orange (★) = Supriya Sule.",
      theme    = theme(plot.title    = element_text(face = "bold", colour = NAVY, size = 13),
                       plot.subtitle = element_text(size = 10, colour = "grey40"))
    )
} else {
  (p_bench_q | p_bench_hhi | p_bench_cs) +
    plot_annotation(
      title    = "Supriya Sule vs Maharashtra MP peers",
      subtitle = "Ranked on three metrics. Orange (★) = Supriya Sule.",
      theme    = theme(plot.title    = element_text(face = "bold", colour = NAVY, size = 13),
                       plot.subtitle = element_text(size = 10, colour = "grey40"))
    )
}

ggsave(file.path(FIGDIR, "sule_peer_benchmark.png"),
       p_bench, width = 13, height = 9, dpi = 180)
cat("Saved: sule_peer_benchmark.png\n")
#}

cat("\n=== E1 complete ===\n")
cat("All Supriya Sule figures saved to:", FIGDIR, "\n")
