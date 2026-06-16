# =============================================================================
# N1_rs_analysis.R
# Author: Piyush Zaware
# Updated: 2026-06-16
#
# Goal: Run all RS-specific analyses (discipline, ministry, sentiment, vocabulary)
#       needed for the LS vs RS comparison page.
#
# Inputs:
#   tmp/rajyasabha_clean.parquet
#   input/rs_name_crosswalk.csv
#
# Outputs:
#   output/tables/rs_discipline_scores.csv
#   output/tables/rs_ministry_party_counts.csv
#   output/tables/rs_ministry_excess.csv
#   output/tables/rs_sentiment_party.csv
#   output/tables/rs_word_freq.csv
#   output/figures/rs_discipline_party.png
#   output/figures/rs_ministry_heatmap.png
#   output/figures/rs_ministry_excess_heatmap.png
#   output/figures/rs_sentiment_party.png
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
  for (pkg in c("textdata", "proxy", "Matrix")) {
    if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  }
  library(proxy)
  library(Matrix)
})

PARTY_COLORS <- c(
  BJP = "#FF6B35", INC = "#1565C0", Left = "#B71C1C", TMC = "#1A237E",
  SP  = "#E53935", BSP = "#6A1B9A", DMK = "#004D40", BJD = "#00695C",
  TDP = "#F9A825", TRS = "#00838F", JDU = "#2E7D32", AIADMK = "#AD1457",
  NCP = "#4527A0", AAP = "#00BCD4", RJD = "#FF8F00", Other = "#78909C"
)

cat("[N1] Loading RS data...\n")

# =============================================================================
# SECTION 1: Load RS starred questions with party crosswalk
# =============================================================================
#{
rs_raw <- read_parquet(file.path(TMPDIR, "rajyasabha_clean.parquet")) %>%
  mutate(
    qtype = str_to_upper(str_trim(qtype)),
    year  = as.integer(str_sub(as.character(adate), 1, 4))
  ) %>%
  filter(qtype == "STARRED", year >= 2014) %>%
  mutate(
    raw_name      = str_squish(replace_na(as.character(name), "")),
    question_text = str_squish(replace_na(as.character(qtitle), "")),
    ministry_raw  = str_squish(str_to_upper(replace_na(as.character(min_name), "")))
  ) %>%
  filter(nchar(raw_name) > 1, nchar(question_text) > 3)

cw <- read_csv(file.path(INPDIR, "rs_name_crosswalk.csv"), show_col_types = FALSE)

rs <- rs_raw %>%
  left_join(cw %>% select(raw_name, party_family), by = "raw_name") %>%
  filter(!is.na(party_family))

cat(sprintf("  RS starred matched: %d / %d (%.1f%%)\n",
            nrow(rs), nrow(rs_raw), 100 * nrow(rs) / nrow(rs_raw)))

# Standardise ministry names (collapse variant spellings)
ministry_map <- c(
  "AGRICULTURE AND FARMERS WELFARE" = "Agriculture",
  "AGRICULTURE"                     = "Agriculture",
  "CHEMICALS AND FERTILIZERS"       = "Chemicals & Fertilizers",
  "CIVIL AVIATION"                  = "Civil Aviation",
  "COAL"                            = "Coal & Mines",
  "MINES"                           = "Coal & Mines",
  "COMMERCE AND INDUSTRY"           = "Commerce & Industry",
  "COMMERCE AND INDUSTRY   "        = "Commerce & Industry",
  "COMMUNICATION AND INFORMATION TECHNOLOGY" = "Communications & IT",
  "COMMUNICATIONS"                  = "Communications & IT",
  "ELECTRONICS AND INFORMATION TECHNOLOGY"   = "Communications & IT",
  "CONSUMER AFFAIRS, FOOD AND PUBLIC DISTRIBUTION" = "Consumer Affairs",
  "COOPERATION"                     = "Cooperation",
  "CORPORATE AFFAIRS"               = "Corporate Affairs",
  "CULTURE"                         = "Culture",
  "DEFENCE"                         = "Defence",
  "DEVELOPMENT OF NORTH EASTERN REGION"      = "North East Development",
  "DEVELOPMENT OF NORTH EASTERN REGION   "   = "North East Development",
  "DRINKING WATER AND SANITATION"   = "Jal Shakti / Water",
  "JAL SHAKTI"                      = "Jal Shakti / Water",
  "WATER RESOURCES"                 = "Jal Shakti / Water",
  "EARTH SCIENCES"                  = "Earth Sciences",
  "EARTH SCIENCES "                 = "Earth Sciences",
  "EDUCATION"                       = "Education",
  "EXTERNAL AFFAIRS"                = "External Affairs",
  "FINANCE"                         = "Finance",
  "FISHERIES"                       = "Fisheries & Animal Husbandry",
  "FISHERIES, ANIMAL HUSBANDRY AND DAIRYING" = "Fisheries & Animal Husbandry",
  "FOOD PROCESSING INDUSTRIES"      = "Food Processing",
  "HEALTH AND FAMILY WELFARE"       = "Health",
  "HEAVY INDUSTRIES"                = "Heavy Industries",
  "HOME AFFAIRS"                    = "Home Affairs",
  "HOUSING AND URBAN AFFAIRS"       = "Housing & Urban Affairs",
  "INFORMATION AND BROADCASTING"    = "Information & Broadcasting",
  "LABOUR AND EMPLOYMENT"           = "Labour & Employment",
  "LAW AND JUSTICE"                 = "Law & Justice",
  "MICRO, SMALL AND MEDIUM ENTERPRISES" = "MSME",
  "MINORITY AFFAIRS"                = "Minority Affairs",
  "NEW AND RENEWABLE ENERGY"        = "Petroleum & Energy",
  "OIL"                             = "Petroleum & Energy",
  "PETROLEUM AND NATURAL GAS"       = "Petroleum & Energy",
  "POWER"                           = "Petroleum & Energy",
  "PORTS, SHIPPING AND WATERWAYS"   = "Ports & Shipping",
  "RAILWAYS"                        = "Railways",
  "ROAD TRANSPORT AND HIGHWAYS"     = "Road Transport",
  "RURAL DEVELOPMENT"               = "Rural Development",
  "SCIENCE AND TECHNOLOGY"          = "Science & Technology",
  "SHIPPING"                        = "Ports & Shipping",
  "SOCIAL JUSTICE AND EMPOWERMENT"  = "Social Justice",
  "SPORTS"                          = "Sports & Youth",
  "YOUTH AFFAIRS AND SPORTS"        = "Sports & Youth",
  "STATISTICS AND PROGRAMME IMPLEMENTATION" = "Statistics",
  "STEEL"                           = "Steel",
  "TEXTILES"                        = "Textiles",
  "TOURISM"                         = "Tourism",
  "TRIBAL AFFAIRS"                  = "Tribal Affairs",
  "WOMEN AND CHILD DEVELOPMENT"     = "Women & Child"
)

rs <- rs %>%
  mutate(ministry = coalesce(ministry_map[ministry_raw], str_to_title(ministry_raw))) %>%
  filter(nchar(ministry) > 1)

cat(sprintf("  Unique parties: %d | Unique ministries: %d\n",
            n_distinct(rs$party_family), n_distinct(rs$ministry)))
#}

# =============================================================================
# SECTION 2: RS Party Discipline
# =============================================================================
#{
cat("[N1] Computing RS party discipline...\n")

MIN_Q <- 5  # minimum starred questions for inclusion

# Build TF-IDF per RS member
rs_tfidf <- rs %>%
  select(raw_name, party_family, question_text) %>%
  unnest_tokens(word, question_text) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2, !str_detect(word, "^[0-9]+$")) %>%
  count(raw_name, party_family, word) %>%
  bind_tf_idf(word, raw_name, n)

# Member question counts
member_counts <- rs %>% count(raw_name, party_family, name = "n_q")

eligible <- member_counts %>% filter(n_q >= MIN_Q) %>% pull(raw_name)

rs_tfidf_filt <- rs_tfidf %>% filter(raw_name %in% eligible)

# Cosine similarity to party centroid
compute_rs_discipline <- function(party) {
  members <- rs_tfidf_filt %>% filter(party_family == party)
  if (n_distinct(members$raw_name) < 2) return(NULL)

  # Sparse doc × word matrix
  docs  <- unique(members$raw_name)
  words <- unique(members$word)
  r <- match(members$raw_name, docs)
  c <- match(members$word,     words)
  mat <- sparseMatrix(i = r, j = c, x = members$tf_idf,
                      dims = c(length(docs), length(words)))

  centroid <- colMeans(mat)
  sims <- as.numeric(proxy::simil(as.matrix(mat),
                                  matrix(centroid, nrow = 1), method = "cosine"))
  tibble(party_family = party, n_mps = length(docs),
         discipline = mean(sims, na.rm = TRUE),
         discipline_sd = sd(sims, na.rm = TRUE))
}

rs_disc <- map_dfr(unique(rs_tfidf_filt$party_family), compute_rs_discipline)
write_csv(rs_disc, file.path(TABDIR, "rs_discipline_scores.csv"))
cat(sprintf("  Saved rs_discipline_scores.csv (%d parties)\n", nrow(rs_disc)))

# Figure: RS discipline bar chart
parties_to_plot <- rs_disc %>% filter(n_mps >= 3) %>% arrange(desc(discipline))

p <- ggplot(parties_to_plot, aes(x = reorder(party_family, discipline),
                                  y = discipline,
                                  fill = party_family)) +
  geom_col(show.legend = FALSE, width = 0.7) +
  geom_errorbar(aes(ymin = pmax(0, discipline - discipline_sd),
                    ymax = discipline + discipline_sd),
                width = 0.3, colour = "grey40", linewidth = 0.4) +
  geom_text(aes(label = sprintf("n=%d", n_mps), y = 0.01),
            hjust = 0, size = 3, colour = "white", fontface = "bold") +
  scale_fill_manual(values = PARTY_COLORS, na.value = "#78909C") +
  coord_flip() +
  labs(title = "Rajya Sabha: party discipline scores",
       subtitle = "Mean cosine similarity of each member to their party centroid",
       x = NULL, y = "Discipline score") +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "rs_discipline_party.png"), p,
       width = 7, height = 5, dpi = 150)
cat("  Saved rs_discipline_party.png\n")
#}

# =============================================================================
# SECTION 3: RS Ministry Analysis
# =============================================================================
#{
cat("[N1] Computing RS ministry targeting...\n")

TOP_MIN <- 20

party_total <- rs %>% count(party_family, name = "party_n")
ministry_total <- rs %>% count(ministry, name = "min_n") %>%
  arrange(desc(min_n)) %>% slice_head(n = TOP_MIN)

top_ministries <- ministry_total$ministry

pm_counts <- rs %>%
  filter(ministry %in% top_ministries) %>%
  count(party_family, ministry) %>%
  left_join(party_total, by = "party_family") %>%
  left_join(ministry_total, by = "ministry") %>%
  mutate(
    expected = party_n * min_n / nrow(rs),
    pearson  = (n - expected) / sqrt(expected)
  )

write_csv(pm_counts, file.path(TABDIR, "rs_ministry_party_counts.csv"))
write_csv(pm_counts %>% select(party_family, ministry, pearson),
          file.path(TABDIR, "rs_ministry_excess.csv"))
cat("  Saved rs_ministry tables\n")

# Focus on parties with enough questions
party_order <- party_total %>% filter(party_n >= 50) %>% arrange(desc(party_n)) %>% pull(party_family)
pm_plot <- pm_counts %>% filter(party_family %in% party_order)

# Ministry heatmap (proportion)
pm_prop <- pm_plot %>%
  mutate(prop = n / party_n) %>%
  select(party_family, ministry, prop)

p2 <- ggplot(pm_prop, aes(x = party_family, y = reorder(ministry, -prop),
                            fill = prop * 100)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  scale_fill_gradient(low = "#EEF2FF", high = "#1A237E",
                      name = "% of party\nquestions") +
  labs(title = "Rajya Sabha: ministry targeting by party",
       subtitle = "% of each party's starred questions directed at each ministry",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "rs_ministry_heatmap.png"), p2,
       width = 9, height = 7, dpi = 150)

# Excess heatmap (Pearson residuals)
p3 <- ggplot(pm_plot %>% filter(party_family %in% party_order),
             aes(x = party_family, y = reorder(ministry, pearson),
                 fill = pmin(pmax(pearson, -4), 4))) +
  geom_tile(colour = "white", linewidth = 0.3) +
  geom_text(data = pm_plot %>% filter(abs(pearson) > 1.5, party_family %in% party_order),
            aes(label = round(pearson, 1)), size = 2.5, colour = "white", fontface = "bold") +
  scale_fill_gradient2(low = "#1565C0", mid = "white", high = "#B71C1C",
                       midpoint = 0, name = "Pearson\nresidual") +
  labs(title = "Rajya Sabha: excess ministry targeting",
       subtitle = "Red = over-questions; blue = under-questions (relative to party size)",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "rs_ministry_excess_heatmap.png"), p3,
       width = 9, height = 7, dpi = 150)
cat("  Saved rs_ministry_heatmap.png + rs_ministry_excess_heatmap.png\n")
#}

# =============================================================================
# SECTION 4: RS Sentiment / Adversarialism (AFINN)
# =============================================================================
#{
cat("[N1] Computing RS adversarialism (BING lexicon)...\n")

bing <- get_sentiments("bing")

rs_sent_q <- rs %>%
  select(raw_name, party_family, qno, adate, question_text) %>%
  unnest_tokens(word, question_text) %>%
  inner_join(bing, by = "word") %>%
  count(raw_name, party_family, qno, adate, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0L) %>%
  mutate(score = positive - negative, is_adversarial = negative > positive)

rs_sent_party <- rs_sent_q %>%
  group_by(party_family) %>%
  summarise(
    n_questions     = n(),
    mean_score      = mean(score, na.rm = TRUE),
    pct_adversarial = 100 * mean(is_adversarial, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(pct_adversarial)

write_csv(rs_sent_party, file.path(TABDIR, "rs_sentiment_party.csv"))
cat("  Saved rs_sentiment_party.csv\n")

p4 <- ggplot(rs_sent_party %>% filter(n_questions >= 20),
             aes(x = reorder(party_family, pct_adversarial),
                 y = pct_adversarial, fill = party_family)) +
  geom_col(show.legend = FALSE, width = 0.7) +
  scale_fill_manual(values = PARTY_COLORS, na.value = "#78909C") +
  coord_flip() +
  labs(title = "Rajya Sabha: adversarialism by party (AFINN)",
       subtitle = "% of starred question titles with negative sentiment",
       x = NULL, y = "% adversarial") +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "rs_sentiment_party.png"), p4,
       width = 7, height = 5, dpi = 150)
cat("  Saved rs_sentiment_party.png\n")
#}

# =============================================================================
# SECTION 5: RS Vocabulary (word frequencies for comparison)
# =============================================================================
#{
cat("[N1] Building RS word frequency table...\n")

rs_words <- rs %>%
  unnest_tokens(word, question_text) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2, !str_detect(word, "^[0-9]+$")) %>%
  count(word, sort = TRUE) %>%
  mutate(freq_rs = n / sum(n))

write_csv(rs_words, file.path(TABDIR, "rs_word_freq.csv"))
cat(sprintf("  Saved rs_word_freq.csv (%d unique words)\n", nrow(rs_words)))
#}

cat("\n[N1] Done.\n")
