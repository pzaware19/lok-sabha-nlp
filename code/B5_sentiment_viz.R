# =============================================================================
# B5_sentiment_viz.R — Sentiment Analysis Visualization
# Author: Piyush Zaware
# Updated: 2026-06-13
#
# PURPOSE:
#   Visualizes VADER sentiment outputs from B3.
#   Key research question: do opposition parties ask more adversarial
#   questions than government parties? Does adversarialism change over time?
#
# INPUTS:
#   output/tables/sentiment_doc.csv
#   output/tables/sentiment_party.csv
#   output/tables/sentiment_ministry.csv
#   output/tables/sentiment_temporal.csv
#   output/tables/sentiment_topic.csv
#   output/tables/bertopic_topic_words.csv  (for topic labels)
#
# OUTPUTS:
#   output/figures/sentiment_party.png
#   output/figures/sentiment_ministry.png
#   output/figures/sentiment_temporal.png
#   output/figures/sentiment_topic.png
#   output/figures/sentiment_distribution.png
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(ggrepel)
})

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")

# Party colors — consistent across all plots
PARTY_COLORS <- c(
  BJP        = "#FF6B35",
  INC        = "#1F78B4",
  Left       = "#E31A1C",
  BSP        = "#6A3D9A",
  AAP        = "#33A02C",
  TMC        = "#B15928",
  SP         = "#FF7F00",
  JDU        = "#A6CEE3",
  DMK        = "#B2DF8A",
  TDP        = "#FDBF6F",
  `Shiv Sena`= "#FB9A99",
  RJD        = "#CAB2D6",
  NCP        = "#999999",
  Regional   = "#CCCCCC"
)

# ============================================================
# SECTION 1: Party-level sentiment bar chart
# ============================================================
#{

party_sent <- read_csv(file.path(TABDIR, "sentiment_party.csv"),
                       show_col_types = FALSE) %>%
  filter(n_questions >= 20)

# Government parties (16th–18th LS)
GOV_PARTIES <- c("BJP", "JDU", "TDP", "Shiv Sena", "NCP")

party_sent <- party_sent %>%
  mutate(
    gov_status = if_else(party_family %in% GOV_PARTIES, "Government", "Opposition"),
    fill_col   = PARTY_COLORS[party_family],
    fill_col   = if_else(is.na(fill_col), "#AAAAAA", fill_col)
  )

p_party_sent <- ggplot(party_sent,
                       aes(x = reorder(party_family, mean_compound),
                           y = mean_compound,
                           fill = party_family)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_errorbar(aes(ymin = mean_compound - sd_compound / sqrt(n_questions),
                    ymax = mean_compound + sd_compound / sqrt(n_questions)),
                width = 0.25, linewidth = 0.5, colour = "grey40") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  scale_fill_manual(values = PARTY_COLORS, na.value = "#AAAAAA") +
  scale_y_continuous(labels = scales::number_format(accuracy = 0.01)) +
  coord_flip() +
  labs(
    title    = "Question tone by party: VADER compound sentiment score",
    subtitle = "Higher = more positive/aspirational; lower = more adversarial/critical.\nError bars = ±1 SE. Only parties with 20+ matched starred questions.",
    x = NULL, y = "Mean VADER compound score",
    caption  = "VADER is calibrated for informal English; formal parliamentary language inflates positive scores.\n Relative differences across parties are interpretable, not absolute levels."
  ) +
  facet_wrap(~ gov_status, scales = "free_y") +
  theme_minimal(base_size = 11) +
  theme(
    plot.title  = element_text(face = "bold"),
    strip.text  = element_text(face = "bold"),
    axis.text.y = element_text(size = 10)
  )

ggsave(file.path(FIGDIR, "sentiment_party.png"),
       p_party_sent, width = 11, height = 6, dpi = 180)
cat("Saved: sentiment_party.png\n")
#}

# ============================================================
# SECTION 2: Adversarialism rate by party (stacked bar)
# ============================================================
#{

party_order_adv <- party_sent %>% arrange(desc(pct_adversarial)) %>% pull(party_family)

p_adv <- party_sent %>%
  select(party_family, pct_adversarial, pct_neutral, pct_positive, n_questions) %>%
  pivot_longer(c(pct_adversarial, pct_neutral, pct_positive),
               names_to = "tone", values_to = "pct") %>%
  mutate(
    tone = recode(tone,
                  pct_adversarial = "Adversarial (compound ≤ −0.05)",
                  pct_neutral     = "Neutral",
                  pct_positive    = "Aspirational/Positive"),
    tone = factor(tone, levels = c("Aspirational/Positive", "Neutral",
                                   "Adversarial (compound ≤ −0.05)")),
    party_family = factor(party_family, levels = party_order_adv)
  ) %>%
  ggplot(aes(x = party_family, y = pct, fill = tone)) +
  geom_col(width = 0.75, position = "stack") +
  scale_fill_manual(
    values = c(
      "Aspirational/Positive"         = "#4DAF4A",
      "Neutral"                       = "#CCCCCC",
      "Adversarial (compound ≤ −0.05)"= "#E31A1C"
    ),
    name = NULL
  ) +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  labs(
    title    = "Share of adversarial vs aspirational questions by party",
    subtitle = "Adversarial = VADER compound ≤ −0.05 ('has the government failed...', 'what steps were not taken...')",
    x = NULL, y = "% of party's starred questions",
    caption  = paste0("n = ", sum(party_sent$n_questions), " matched starred questions (38.6% of total).")
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title      = element_text(face = "bold"),
    axis.text.x     = element_text(angle = 30, hjust = 1)
  )

ggsave(file.path(FIGDIR, "sentiment_distribution.png"),
       p_adv, width = 11, height = 6, dpi = 180)
cat("Saved: sentiment_distribution.png\n")
#}

# ============================================================
# SECTION 3: Temporal — adversarialism by party × Lok Sabha
# ============================================================
#{

temporal_sent <- read_csv(file.path(TABDIR, "sentiment_temporal.csv"),
                          show_col_types = FALSE) %>%
  filter(n_questions >= 10) %>%
  mutate(lok_label = paste0(lok_no, "th LS"))

major_parties <- c("BJP", "INC", "Left", "Regional", "DMK", "TDP", "JDU")

p_temporal <- temporal_sent %>%
  filter(party_family %in% major_parties) %>%
  ggplot(aes(x = lok_label, y = pct_adversarial,
             group = party_family, colour = party_family)) +
  geom_line(linewidth = 1) +
  geom_point(aes(size = n_questions)) +
  scale_colour_manual(values = PARTY_COLORS, na.value = "#999999", name = NULL) +
  scale_size_continuous(name = "n questions", range = c(2, 7)) +
  labs(
    title    = "Adversarialism over time: have opposition parties become more aggressive?",
    subtitle = "% of starred questions classified as adversarial (VADER compound ≤ −0.05), by Lok Sabha",
    x = NULL, y = "% adversarial questions"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "right",
    plot.title      = element_text(face = "bold")
  )

ggsave(file.path(FIGDIR, "sentiment_temporal.png"),
       p_temporal, width = 10, height = 5.5, dpi = 180)
cat("Saved: sentiment_temporal.png\n")
#}

# ============================================================
# SECTION 4: Ministry-level adversarialism
# ============================================================
#{

ministry_sent <- read_csv(file.path(TABDIR, "sentiment_ministry.csv"),
                          show_col_types = FALSE) %>%
  filter(n_questions >= 50) %>%
  mutate(ministry_short = str_trunc(str_to_title(ministry), 35))

p_ministry <- ggplot(ministry_sent,
                     aes(x = reorder(ministry_short, -pct_adversarial),
                         y = pct_adversarial,
                         fill = mean_compound)) +
  geom_col(width = 0.75) +
  scale_fill_gradient2(
    low = "#B5440E", mid = "#F5F5F5", high = "#1F78B4",
    midpoint = median(ministry_sent$mean_compound),
    name = "Mean\nsentiment"
  ) +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  coord_flip() +
  labs(
    title    = "Which ministries attract the most adversarial questioning?",
    subtitle = "% adversarial starred questions directed at each ministry. Fill = mean compound score.",
    x = NULL, y = "% adversarial questions",
    caption  = "Ministries with 50+ starred questions only."
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title  = element_text(face = "bold"),
    axis.text.y = element_text(size = 9)
  )

ggsave(file.path(FIGDIR, "sentiment_ministry.png"),
       p_ministry, width = 11, height = 8, dpi = 180)
cat("Saved: sentiment_ministry.png\n")
#}

# ============================================================
# SECTION 5: BERTopic topic × sentiment
# ============================================================
#{

topic_sent_path <- file.path(TABDIR, "sentiment_topic.csv")
topic_words_path <- file.path(TABDIR, "bertopic_topic_words.csv")

if (file.exists(topic_sent_path) && file.exists(topic_words_path)) {
  topic_sent  <- read_csv(topic_sent_path,  show_col_types = FALSE)
  topic_words <- read_csv(topic_words_path, show_col_types = FALSE)

  # Get label for each topic (top 3 words)
  topic_labels <- topic_words %>%
    group_by(topic_id) %>%
    slice_max(score, n = 3) %>%
    summarise(label = paste(word, collapse = " / "), .groups = "drop")

  topic_sent <- topic_sent %>%
    left_join(topic_labels, by = "topic_id") %>%
    filter(!is.na(label))

  p_topic_sent <- ggplot(topic_sent,
                         aes(x = mean_compound,
                             y = pct_adversarial,
                             size = n_questions,
                             label = label)) +
    geom_point(colour = "#B5440E", alpha = 0.7) +
    geom_text_repel(size = 2.8, max.overlaps = 20, box.padding = 0.3) +
    scale_size_continuous(name = "n questions", range = c(3, 10)) +
    labs(
      title    = "BERTopic topics: sentiment vs adversarialism",
      subtitle = "Each point = one BERTopic topic cluster. Topics on the left attract more critical questioning.",
      x = "Mean VADER compound score (lower = more critical)",
      y = "% adversarial questions in topic",
      caption  = "Labels = top 3 c-TF-IDF words per topic."
    ) +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))

  ggsave(file.path(FIGDIR, "sentiment_topic.png"),
         p_topic_sent, width = 11, height = 7, dpi = 180)
  cat("Saved: sentiment_topic.png\n")
}
#}

cat("\n=== B5 complete ===\n")
