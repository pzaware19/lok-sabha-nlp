# =============================================================================
# B2_bertopic_viz.R — BERTopic Results Visualization
# Author: Piyush Zaware
# Updated: 2026-06-12
#
# PURPOSE:
#   Visualize BERTopic outputs and compare against LDA.
#   BERTopic advantages over LDA:
#     - Works in semantic embedding space, not bag-of-words
#     - Topic words are contextually coherent, not just co-frequent
#     - Handles boilerplate better (parliamentary preamble)
#     - Finds topics hierarchically via HDBSCAN clustering
#
# INPUTS:
#   output/tables/bertopic_topics.csv
#   output/tables/bertopic_topic_words.csv
#   output/tables/bertopic_doc_assignments.csv
#   output/tables/bertopic_party_matrix.csv
#   output/tables/lda_top_terms.csv           (from A4, for comparison)
#
# OUTPUTS:
#   output/figures/bertopic_top_terms.png
#   output/figures/bertopic_party_heatmap.png
#   output/figures/bertopic_vs_lda.png
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidytext)
  library(patchwork)
  library(ggrepel)
})

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")

# ============================================================
# SECTION 1: Load BERTopic outputs
# ============================================================
#{

topic_info  <- read_csv(file.path(TABDIR, "bertopic_topics.csv"),
                        show_col_types = FALSE)
topic_words <- read_csv(file.path(TABDIR, "bertopic_topic_words.csv"),
                        show_col_types = FALSE)
doc_assign  <- read_csv(file.path(TABDIR, "bertopic_doc_assignments.csv"),
                        show_col_types = FALSE)

cat("Topics (excl. outlier -1):", nrow(topic_info[topic_info$Topic != -1, ]), "\n")
cat("Docs assigned:", nrow(doc_assign), "\n")
cat("Outlier rate:", round(100 * mean(doc_assign$topic_id == -1), 1), "%\n")
#}

# ============================================================
# SECTION 2: Top-terms bar chart per topic
# ============================================================
#{

# Keep top 25 topics by document count
top_topics <- topic_info %>%
  filter(Topic != -1) %>%
  slice_max(Count, n = 25) %>%
  pull(Topic)

# Clean up auto-generated topic name: "0_railway_trains_..." → readable label
topic_labels <- topic_info %>%
  filter(Topic %in% top_topics) %>%
  mutate(
    label = Name %>%
      str_remove("^\\d+_") %>%
      str_replace_all("_", " ") %>%
      str_to_title() %>%
      str_trunc(35)
  ) %>%
  select(Topic, label, Count)

p_terms <- topic_words %>%
  filter(topic_id %in% top_topics) %>%
  group_by(topic_id) %>%
  slice_max(score, n = 8) %>%
  ungroup() %>%
  left_join(topic_labels, by = c("topic_id" = "Topic")) %>%
  mutate(
    label   = paste0(label, "\n(n=", Count, ")"),
    word    = reorder_within(word, score, topic_id)
  ) %>%
  ggplot(aes(x = word, y = score, fill = factor(topic_id))) +
  geom_col(show.legend = FALSE, width = 0.7) +
  coord_flip() +
  facet_wrap(~ label, scales = "free_y", ncol = 5) +
  scale_x_reordered() +
  scale_fill_viridis_d(option = "turbo") +
  labs(
    title    = "BERTopic: top terms per topic (c-TF-IDF scores)",
    subtitle = "25 largest topics. Unlike LDA, BERTopic uses class-based TF-IDF — terms distinctive to each topic cluster.",
    x = NULL, y = "c-TF-IDF score",
    caption  = paste("n = 11,841 starred questions. Outlier rate:",
                     round(100 * mean(doc_assign$topic_id == -1), 1), "%.")
  ) +
  theme_minimal(base_size = 8.5) +
  theme(
    plot.title  = element_text(face = "bold", size = 11),
    strip.text  = element_text(face = "bold", size = 7),
    axis.text.y = element_text(size = 7)
  )

ggsave(file.path(FIGDIR, "bertopic_top_terms.png"),
       p_terms, width = 16, height = 14, dpi = 160)
cat("Saved: bertopic_top_terms.png\n")
#}

# ============================================================
# SECTION 3: Party × topic heatmap (from matched subset)
# ============================================================
#{

party_matrix_path <- file.path(TABDIR, "bertopic_party_matrix.csv")
if (file.exists(party_matrix_path)) {
  party_mat <- read_csv(party_matrix_path, show_col_types = FALSE)

  # Convert wide to long; exclude Unknown
  party_long <- party_mat %>%
    pivot_longer(-party_family, names_to = "topic_id", values_to = "proportion") %>%
    mutate(topic_id = as.integer(topic_id)) %>%
    filter(party_family != "Unknown", topic_id %in% top_topics) %>%
    left_join(topic_labels, by = c("topic_id" = "Topic")) %>%
    filter(!is.na(label))

  if (nrow(party_long) > 0 && n_distinct(party_long$party_family) >= 3) {
    p_party_heatmap <- ggplot(party_long,
                              aes(x = party_family,
                                  y = reorder(label, proportion, sum),
                                  fill = proportion * 100)) +
      geom_tile(colour = "white", linewidth = 0.3) +
      scale_fill_distiller(palette = "YlOrRd", direction = 1,
                           name = "% of\nquestions") +
      labs(
        title    = "BERTopic: party topic profiles",
        subtitle = "Share of each party's matched questions falling in each BERTopic topic cluster.",
        x = NULL, y = NULL,
        caption  = "Matched subset only. Party match rate ~32%."
      ) +
      theme_minimal(base_size = 10) +
      theme(
        axis.text.x  = element_text(angle = 35, hjust = 1),
        plot.title   = element_text(face = "bold"),
        legend.position = "right"
      )

    ggsave(file.path(FIGDIR, "bertopic_party_heatmap.png"),
           p_party_heatmap, width = 12, height = 8, dpi = 180)
    cat("Saved: bertopic_party_heatmap.png\n")
  } else {
    cat("Insufficient matched party data for heatmap.\n")
  }
}
#}

# ============================================================
# SECTION 4: BERTopic vs LDA — topic count and top-term comparison
# ============================================================
#{

lda_terms <- read_csv(file.path(TABDIR, "lda_top_terms.csv"),
                      show_col_types = FALSE) %>%
  group_by(topic) %>%
  slice_max(beta, n = 5) %>%
  summarise(top_words = paste(term, collapse = ", "), .groups = "drop") %>%
  mutate(method = "LDA (K=15)", topic_id = topic)

bert_terms <- topic_words %>%
  filter(topic_id %in% top_topics) %>%
  group_by(topic_id) %>%
  slice_max(score, n = 5) %>%
  summarise(top_words = paste(word, collapse = ", "), .groups = "drop") %>%
  left_join(topic_labels %>% rename(topic_id = Topic), by = "topic_id") %>%
  mutate(method = "BERTopic (K=auto)", topic_id = topic_id) %>%
  select(method, topic_id, top_words)

comparison_df <- bind_rows(
  lda_terms   %>% mutate(n_topics = 15, outlier_pct = 0),
  bert_terms  %>% mutate(n_topics = n_distinct(top_topics),
                          outlier_pct = round(mean(doc_assign$topic_id == -1) * 100, 1))
)

# Summary comparison panel
comparison_summary <- tibble(
  Feature  = c("Method", "Topic count", "Document unit",
                "Vocabulary", "Outlier handling",
                "Topic coherence", "Boilerplate sensitivity"),
  LDA      = c("Dirichlet allocation", "15 (hand-tuned)",
                "Party × session aggregate", "Bag-of-words (stemmed)",
                "None — every doc assigned", "Moderate", "High"),
  BERTopic = c("HDBSCAN + c-TF-IDF", paste0(n_distinct(top_topics), " (auto)"),
               "Individual starred question", "Sentence embeddings (multilingual)",
               paste0(round(mean(doc_assign$topic_id==-1)*100,1), "% as outliers"),
               "High", "Low")
)

p_compare <- comparison_summary %>%
  pivot_longer(c(LDA, BERTopic), names_to = "Method", values_to = "Value") %>%
  ggplot(aes(x = Method, y = fct_rev(factor(Feature)), label = Value,
             fill = Method)) +
  geom_tile(colour = "white", linewidth = 0.5, alpha = 0.15) +
  geom_text(size = 3, lineheight = 1.1) +
  scale_fill_manual(values = c(LDA = "#1F78B4", BERTopic = "#B5440E"),
                    guide = "none") +
  labs(
    title    = "LDA vs BERTopic: method comparison",
    subtitle = "BERTopic works in semantic space; LDA in token space. Both are run on the same corpus.",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title  = element_text(face = "bold"),
    axis.text.x = element_text(face = "bold", size = 11),
    panel.grid  = element_blank()
  )

ggsave(file.path(FIGDIR, "bertopic_vs_lda.png"),
       p_compare, width = 10, height = 5, dpi = 180)
cat("Saved: bertopic_vs_lda.png\n")
#}

cat("\n=== B2 complete ===\n")
