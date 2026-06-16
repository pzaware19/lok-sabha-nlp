# =============================================================================
# N2_compare_LS_RS.R
# Author: Piyush Zaware
# Updated: 2026-06-16
#
# Goal: Build all comparison figures for the LS vs RS comparison page.
#       Requires N1_rs_analysis.R to have run first.
#
# Inputs:
#   output/tables/rs_discipline_scores.csv
#   output/tables/rs_ministry_party_counts.csv
#   output/tables/rs_sentiment_party.csv
#   output/tables/rs_word_freq.csv
#   output/tables/discipline_scores.csv
#   output/tables/ministry_party_counts.csv  (LS)
#   output/tables/sentiment_party.csv        (LS VADER)
#   output/tables/ideal_points.csv           (LS)
#   output/tables/rs_ideal_points.csv        (RS)
#   tmp/rajyasabha_clean.parquet
#   tmp/train-0000[0-4]-of-00005.parquet
#   input/rs_name_crosswalk.csv
#   input/mp_name_crosswalk.csv
#
# Outputs:
#   output/figures/compare_question_volume.png
#   output/figures/compare_vocab_logratio.png
#   output/figures/compare_ministry_focus.png
#   output/figures/compare_discipline.png
#   output/figures/compare_adversarialism.png
#   output/figures/compare_party_fingerprints.png
#   output/figures/compare_summary_panel.png
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

cat("[N2] Loading all inputs...\n")

# =============================================================================
# SECTION 1: Load LS and RS data for question volumes
# =============================================================================
#{
cw_ls <- read_csv(file.path(INPDIR, "mp_name_crosswalk.csv"), show_col_types = FALSE)
cw_rs <- read_csv(file.path(INPDIR, "rs_name_crosswalk.csv"), show_col_types = FALSE)

# LS starred questions (matched)
ls_files <- list.files(TMPDIR, pattern = "train-0000[0-4]-of-00005\\.parquet", full.names = TRUE)
ls_q <- bind_rows(lapply(ls_files, read_parquet)) %>%
  filter(type == "STARRED") %>%
  mutate(
    primary_raw = str_to_upper(str_squish(map_chr(members, ~.x[[1]]))),
    lok_no      = as.integer(lok_no)
  ) %>%
  left_join(cw_ls %>% select(raw_name, lok_no, party_family),
            by = c("primary_raw" = "raw_name", "lok_no")) %>%
  filter(!is.na(party_family))

# RS starred questions (matched)
rs_q <- read_parquet(file.path(TMPDIR, "rajyasabha_clean.parquet")) %>%
  mutate(
    qtype    = str_to_upper(str_trim(qtype)),
    year     = as.integer(str_sub(as.character(adate), 1, 4)),
    raw_name = str_squish(replace_na(as.character(name), ""))
  ) %>%
  filter(qtype == "STARRED", year >= 2014, nchar(raw_name) > 1) %>%
  left_join(cw_rs %>% select(raw_name, party_family), by = "raw_name") %>%
  filter(!is.na(party_family))

cat(sprintf("  LS matched: %d | RS matched: %d\n", nrow(ls_q), nrow(rs_q)))

# Question volume by party × house
ls_vol <- ls_q %>% count(party_family, name = "n") %>% mutate(house = "Lok Sabha")
rs_vol <- rs_q %>% count(party_family, name = "n") %>% mutate(house = "Rajya Sabha")

vol <- bind_rows(ls_vol, rs_vol) %>%
  filter(party_family %in% c("BJP","INC","Left","TMC","SP","BSP","DMK","BJD",
                              "TDP","JDU","AIADMK","NCP","AAP","TRS","RJD"))
#}

# =============================================================================
# SECTION 2: Figure 1 - Question volume by party × house
# =============================================================================
#{
cat("[N2] Figure 1: question volume by party × house...\n")

party_order_vol <- vol %>%
  group_by(party_family) %>% summarise(tot = sum(n)) %>%
  arrange(desc(tot)) %>% pull(party_family)

p1 <- ggplot(vol, aes(x = factor(party_family, levels = rev(party_order_vol)),
                       y = n, fill = house)) +
  geom_col(position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("Lok Sabha" = "#1A5276", "Rajya Sabha" = "#B7950B"),
                    name = NULL) +
  coord_flip() +
  labs(
    title    = "Starred question volume: Lok Sabha vs Rajya Sabha",
    subtitle = "Total matched starred questions per party, 2014-2025",
    x = NULL, y = "Starred questions"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top", panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "compare_question_volume.png"), p1,
       width = 8, height = 6, dpi = 150)
cat("  Saved compare_question_volume.png\n")
#}

# =============================================================================
# SECTION 3: Figure 2 - Vocabulary log-ratio (LS vs RS)
# =============================================================================
#{
cat("[N2] Figure 2: vocabulary log-ratio LS vs RS...\n")

# LS word frequencies from question subjects
ls_words <- ls_q %>%
  select(subject) %>%
  unnest_tokens(word, subject) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2, !str_detect(word, "^[0-9]+$")) %>%
  count(word, sort = TRUE) %>%
  mutate(freq_ls = n / sum(n))

rs_words <- read_csv(file.path(TABDIR, "rs_word_freq.csv"), show_col_types = FALSE)

# Join and compute log-ratio
vocab_compare <- ls_words %>%
  rename(n_ls = n) %>%
  full_join(rs_words %>% rename(n_rs = n), by = "word") %>%
  mutate(
    n_ls     = replace_na(n_ls, 0),
    n_rs     = replace_na(n_rs, 0),
    freq_ls  = replace_na(freq_ls, 0),
    freq_rs  = replace_na(freq_rs, 0),
    # smoothed log-ratio
    log_ratio = log2((freq_rs + 1e-6) / (freq_ls + 1e-6)),
    total     = n_ls + n_rs
  ) %>%
  filter(total >= 15)

# Top RS-distinctive and LS-distinctive words
top_rs <- vocab_compare %>% arrange(desc(log_ratio)) %>% head(20)
top_ls <- vocab_compare %>% arrange(log_ratio)      %>% head(20)
logratio_plot <- bind_rows(top_rs, top_ls) %>%
  distinct(word, .keep_all = TRUE) %>%
  mutate(direction = ifelse(log_ratio > 0, "More in RS", "More in LS"))

p2 <- ggplot(logratio_plot,
             aes(x = reorder(word, log_ratio), y = log_ratio,
                 fill = direction)) +
  geom_col(show.legend = FALSE, width = 0.75) +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  scale_fill_manual(values = c("More in RS" = "#B7950B", "More in LS" = "#1A5276")) +
  coord_flip() +
  labs(
    title    = "Vocabulary differences between houses",
    subtitle = "log2(RS frequency / LS frequency). Words used more in Rajya Sabha vs Lok Sabha questions.",
    x = NULL, y = "log2 ratio (positive = more RS)"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "compare_vocab_logratio.png"), p2,
       width = 8, height = 7, dpi = 150)
cat("  Saved compare_vocab_logratio.png\n")
#}

# =============================================================================
# SECTION 4: Figure 3 - Ministry focus comparison
# =============================================================================
#{
cat("[N2] Figure 3: ministry focus comparison LS vs RS...\n")

ls_min_raw <- read_csv(file.path(TABDIR, "ministry_party_counts.csv"), show_col_types = FALSE)
rs_min     <- read_csv(file.path(TABDIR, "rs_ministry_party_counts.csv"), show_col_types = FALSE)

key_parties  <- c("BJP", "INC", "Left", "TMC", "DMK", "AIADMK")
key_ministry <- c("Finance", "Home Affairs", "Railways", "Road Transport",
                  "Health", "Education", "Agriculture", "Labour & Employment",
                  "Defence", "Rural Development", "Housing & Urban Affairs",
                  "Tribal Affairs", "Minority Affairs", "Agriculture",
                  "Jal Shakti / Water", "Women & Child")

# Derive party_n from the ministry file itself so denominator matches the numerator
ls_party_total <- ls_min_raw %>%
  group_by(party_family) %>%
  summarise(party_n = sum(n), .groups = "drop")

ls_min_clean <- ls_min_raw %>%
  rename(ministry = ministry_clean) %>%
  left_join(ls_party_total, by = "party_family") %>%
  mutate(house = "Lok Sabha") %>%
  filter(party_family %in% key_parties, ministry %in% key_ministry, n > 0) %>%
  group_by(party_family, ministry, house, party_n) %>%
  summarise(n = sum(n), .groups = "drop") %>%
  mutate(prop = n / party_n)

rs_min_clean <- rs_min %>%
  mutate(house = "Rajya Sabha") %>%
  filter(party_family %in% key_parties, ministry %in% key_ministry) %>%
  group_by(party_family, ministry, house) %>%
  summarise(n = sum(n), party_n = sum(party_n), .groups = "drop") %>%
  mutate(prop = n / party_n)

min_compare <- bind_rows(ls_min_clean, rs_min_clean)

p3 <- ggplot(min_compare,
             aes(x = party_family, y = reorder(ministry, prop),
                 fill = prop * 100)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = ifelse(prop * 100 >= 3,
                               sprintf("%.0f%%", prop * 100), "")),
            size = 2.5, colour = "white", fontface = "bold") +
  scale_fill_gradient(low = "#EEF2FF", high = "#1A237E",
                      name = "% of party\nquestions") +
  facet_wrap(~house, nrow = 1) +
  labs(title = "Ministry focus: Lok Sabha vs Rajya Sabha",
       subtitle = "% of each party's starred questions to each ministry",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold", size = 11))

ggsave(file.path(FIGDIR, "compare_ministry_focus.png"), p3,
       width = 12, height = 7, dpi = 150)
cat("  Saved compare_ministry_focus.png\n")
#}

# =============================================================================
# SECTION 5: Figure 4 - Party discipline LS vs RS
# =============================================================================
#{
cat("[N2] Figure 4: discipline comparison LS vs RS...\n")

ls_disc <- read_csv(file.path(TABDIR, "discipline_scores.csv"), show_col_types = FALSE) %>%
  group_by(party_family) %>%
  summarise(discipline = mean(discipline, na.rm = TRUE),
            n_mps = sum(n_mps, na.rm = TRUE), .groups = "drop") %>%
  mutate(house = "Lok Sabha")

rs_disc <- read_csv(file.path(TABDIR, "rs_discipline_scores.csv"), show_col_types = FALSE) %>%
  mutate(house = "Rajya Sabha")

disc_compare <- bind_rows(
  ls_disc %>% select(party_family, discipline, n_mps, house),
  rs_disc  %>% select(party_family, discipline, n_mps, house)
) %>%
  filter(party_family %in% c("BJP","INC","Left","TMC","SP","BSP","DMK",
                              "BJD","TDP","JDU","AIADMK"),
         n_mps >= 3)

# Joined to show matched pairs
disc_wide <- disc_compare %>%
  pivot_wider(names_from = house, values_from = c(discipline, n_mps)) %>%
  filter(!is.na(`discipline_Lok Sabha`), !is.na(`discipline_Rajya Sabha`)) %>%
  rename(ls = `discipline_Lok Sabha`, rs = `discipline_Rajya Sabha`)

disc_long <- disc_compare %>%
  filter(party_family %in% disc_wide$party_family)

p4 <- ggplot(disc_long,
             aes(x = reorder(party_family, discipline),
                 y = discipline, colour = house, group = house)) +
  geom_point(aes(size = n_mps), alpha = 0.9) +
  geom_line(data = disc_wide %>%
              pivot_longer(c(ls, rs), names_to = "house", values_to = "discipline") %>%
              mutate(house = ifelse(house == "ls", "Lok Sabha", "Rajya Sabha")),
            aes(x = party_family, y = discipline, group = party_family),
            colour = "grey60", linewidth = 0.5, linetype = "dashed") +
  scale_colour_manual(values = c("Lok Sabha" = "#1A5276", "Rajya Sabha" = "#B7950B"),
                      name = NULL) +
  scale_size_continuous(range = c(3, 10), name = "Members") +
  coord_flip() +
  labs(
    title    = "Party discipline: Lok Sabha vs Rajya Sabha",
    subtitle = "Mean cosine similarity of each member to party centroid. Higher = more coordinated vocabulary.",
    x = NULL, y = "Discipline score"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top", panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "compare_discipline.png"), p4,
       width = 8, height = 6, dpi = 150)
cat("  Saved compare_discipline.png\n")
#}

# =============================================================================
# SECTION 6: Figure 5 - Adversarialism comparison LS vs RS
# =============================================================================
#{
cat("[N2] Figure 5: adversarialism comparison...\n")

# Compute LS adversarialism using BING lexicon (same as RS for consistency)
bing <- get_sentiments("bing")

ls_sent_q <- ls_q %>%
  select(party_family, subject) %>%
  filter(!is.na(subject)) %>%
  mutate(qid = row_number()) %>%
  unnest_tokens(word, subject) %>%
  inner_join(bing, by = "word") %>%
  count(party_family, qid, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n,
              values_fill = 0L, names_expand = TRUE) %>%
  mutate(positive = if ("positive" %in% names(.)) .data$positive else 0L,
         negative = if ("negative" %in% names(.)) .data$negative else 0L,
         score = positive - negative, is_adversarial = negative > positive)

ls_sent_party <- ls_sent_q %>%
  group_by(party_family) %>%
  summarise(n = n(), pct_adversarial = 100 * mean(is_adversarial, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(house = "Lok Sabha")

rs_sent_party <- read_csv(file.path(TABDIR, "rs_sentiment_party.csv"),
                          show_col_types = FALSE) %>%
  mutate(house = "Rajya Sabha") %>%
  rename(n = n_questions)

sent_compare <- bind_rows(
  ls_sent_party %>% select(party_family, n, pct_adversarial, house),
  rs_sent_party %>% select(party_family, n, pct_adversarial, house)
) %>%
  filter(party_family %in% c("BJP","INC","Left","TMC","SP","BSP","DMK",
                              "BJD","TDP","JDU","AIADMK","AAP"),
         n >= 20)

sent_wide <- sent_compare %>%
  select(party_family, house, pct_adversarial, n) %>%
  pivot_wider(names_from = house, values_from = c(pct_adversarial, n),
              names_sep = "_") %>%
  drop_na()

ls_col <- names(sent_wide)[str_detect(names(sent_wide), "pct.*Lok")]
rs_col <- names(sent_wide)[str_detect(names(sent_wide), "pct.*Rajya")]

p5 <- ggplot(sent_compare %>% filter(party_family %in% sent_wide$party_family),
             aes(x = reorder(party_family, pct_adversarial),
                 y = pct_adversarial, colour = house, group = house)) +
  geom_point(aes(size = n), alpha = 0.9) +
  geom_line(data = sent_wide %>%
              pivot_longer(all_of(c(ls_col, rs_col)),
                           names_to = "house", values_to = "pct_adversarial") %>%
              mutate(house = if_else(str_detect(house, "Lok"), "Lok Sabha", "Rajya Sabha")),
            aes(x = party_family, y = pct_adversarial, group = party_family),
            colour = "grey60", linewidth = 0.5, linetype = "dashed") +
  scale_colour_manual(values = c("Lok Sabha" = "#1A5276", "Rajya Sabha" = "#B7950B"),
                      name = NULL) +
  scale_size_continuous(range = c(3, 8), name = "Questions") +
  coord_flip() +
  labs(
    title    = "Adversarialism: Lok Sabha vs Rajya Sabha",
    subtitle = "% of starred question titles with negative sentiment (BING lexicon)",
    x = NULL, y = "% adversarial questions"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top", panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "compare_adversarialism.png"), p5,
       width = 8, height = 6, dpi = 150)
cat("  Saved compare_adversarialism.png\n")
#}

# =============================================================================
# SECTION 7: Figure 6 - Party fingerprints (BJP and INC: LS vs RS vocabulary)
# =============================================================================
#{
cat("[N2] Figure 6: party vocabulary fingerprints LS vs RS...\n")

compute_party_fingerprint <- function(party_name) {
  ls_party <- ls_q %>%
    filter(party_family == party_name) %>%
    unnest_tokens(word, subject) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) > 2, !str_detect(word, "^[0-9]+$")) %>%
    count(word) %>%
    mutate(freq = n / sum(n), house = "Lok Sabha")

  rs_party <- rs_q %>%
    filter(party_family == party_name) %>%
    mutate(question_text = str_squish(replace_na(as.character(qtitle), ""))) %>%
    filter(nchar(question_text) > 2) %>%
    unnest_tokens(word, question_text) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) > 2, !str_detect(word, "^[0-9]+$")) %>%
    count(word) %>%
    mutate(freq = n / sum(n), house = "Rajya Sabha")

  joined <- ls_party %>% rename(freq_ls = freq, n_ls = n) %>%
    full_join(rs_party %>% rename(freq_rs = freq, n_rs = n), by = "word") %>%
    mutate(across(starts_with("freq_"), ~ replace_na(., 0)),
           across(starts_with("n_"),    ~ replace_na(., 0)),
           log_ratio = log2((freq_rs + 1e-6) / (freq_ls + 1e-6)),
           total = n_ls + n_rs) %>%
    filter(total >= 5)

  top_rs_p <- joined %>% top_n(10, log_ratio)
  top_ls_p <- joined %>% top_n(-10, log_ratio)
  bind_rows(top_rs_p, top_ls_p) %>%
    distinct(word, .keep_all = TRUE) %>%
    mutate(party = party_name,
           direction = ifelse(log_ratio > 0, "More RS", "More LS"))
}

fp_data <- bind_rows(
  compute_party_fingerprint("BJP"),
  compute_party_fingerprint("INC"),
  compute_party_fingerprint("Left"),
  compute_party_fingerprint("TMC")
)

p6 <- ggplot(fp_data,
             aes(x = reorder_within(word, log_ratio, party),
                 y = log_ratio, fill = direction)) +
  geom_col(show.legend = FALSE, width = 0.8) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  scale_x_reordered() +
  scale_fill_manual(values = c("More RS" = "#B7950B", "More LS" = "#1A5276")) +
  facet_wrap(~party, scales = "free_y", ncol = 2) +
  coord_flip() +
  labs(
    title    = "Party vocabulary shifts between houses",
    subtitle = "log2 ratio of word frequency in RS vs LS for each party. Orange = used more in RS; blue = used more in LS.",
    x = NULL, y = "log2(RS freq / LS freq)"
  ) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.major.y = element_blank(),
        strip.text = element_text(face = "bold"),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "compare_party_fingerprints.png"), p6,
       width = 10, height = 9, dpi = 150)
cat("  Saved compare_party_fingerprints.png\n")
#}

# =============================================================================
# SECTION 8: Figure 7 - Ideal point comparison (between/within ratio)
# =============================================================================
#{
cat("[N2] Figure 7: between/within variance comparison...\n")

ip_ls <- read_csv(file.path(TABDIR, "ideal_points.csv"),    show_col_types = FALSE)
ip_rs <- read_csv(file.path(TABDIR, "rs_ideal_points.csv"), show_col_types = FALSE)

compute_bw_ratio <- function(ip, house_name) {
  parties <- ip %>% count(party_family) %>% filter(n >= 5) %>% pull(party_family)
  ip_filt <- ip %>% filter(party_family %in% parties)
  grand_mean <- mean(ip_filt$dim1, na.rm = TRUE)
  party_means <- ip_filt %>%
    group_by(party_family) %>%
    summarise(pm = mean(dim1, na.rm = TRUE), n = n(), .groups = "drop")
  between <- sum(party_means$n * (party_means$pm - grand_mean)^2) /
    (nrow(party_means) - 1)
  within  <- ip_filt %>%
    left_join(party_means, by = "party_family") %>%
    mutate(sq = (dim1 - pm)^2) %>%
    pull(sq) %>% mean(na.rm = TRUE)
  tibble(house = house_name, between = between, within = within,
         ratio = between / within, n_members = nrow(ip_filt), n_parties = nrow(party_means))
}

bw <- bind_rows(
  compute_bw_ratio(ip_ls, "Lok Sabha"),
  compute_bw_ratio(ip_rs, "Rajya Sabha")
)

write_csv(bw, file.path(TABDIR, "compare_bw_ratio.csv"))

p7 <- ggplot(bw, aes(x = house, y = ratio, fill = house)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.4f", ratio)), vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("Lok Sabha" = "#1A5276", "Rajya Sabha" = "#B7950B")) +
  annotate("text", x = 1.5, y = max(bw$ratio) * 0.7,
           label = "US NOMINATE\nbetween/within > 10",
           size = 3.5, colour = "grey40", fontface = "italic") +
  labs(
    title    = "Party clustering in question vocabulary: LS vs RS",
    subtitle = "Between-party variance / within-party variance on SVD Dimension 1.\nHigher = more party-based sorting of vocabulary.",
    x = NULL, y = "Between / Within variance ratio"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.x = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "compare_bw_ratio.png"), p7,
       width = 6, height = 5, dpi = 150)
cat("  Saved compare_bw_ratio.png\n")
#}

# =============================================================================
# SECTION 9: Summary statistics table
# =============================================================================
#{
cat("[N2] Building summary statistics table...\n")

ls_matched_n  <- nrow(ls_q)
rs_matched_n  <- nrow(rs_q)
ls_members_n  <- nrow(ip_ls)
rs_members_n  <- nrow(ip_rs)

summary_tbl <- tibble(
  Metric       = c("Starred questions (matched)", "Members / MPs matched",
                   "Houses covered", "Years covered",
                   "Party-vocabulary B/W ratio",
                   "BJP-INC dimension spread (normalised)"),
  `Lok Sabha`  = c(format(ls_matched_n, big.mark=","),
                   format(ls_members_n, big.mark=","),
                   "16th, 17th, 18th Lok Sabha",
                   "2014-2026",
                   sprintf("%.4f", bw$ratio[bw$house=="Lok Sabha"]),
                   sprintf("%.3f", abs(mean(ip_ls$dim1[ip_ls$party_family=="BJP"],na.rm=TRUE) -
                                         mean(ip_ls$dim1[ip_ls$party_family=="INC"],na.rm=TRUE)) /
                     max(abs(mean(ip_ls$dim1[ip_ls$party_family=="BJP"],na.rm=TRUE)),
                         abs(mean(ip_ls$dim1[ip_ls$party_family=="INC"],na.rm=TRUE))))),
  `Rajya Sabha`= c(format(rs_matched_n, big.mark=","),
                   format(rs_members_n, big.mark=","),
                   "2014-2025",
                   "2014-2025",
                   sprintf("%.4f", bw$ratio[bw$house=="Rajya Sabha"]),
                   sprintf("%.3f",
                     abs(mean(ip_rs$dim1[ip_rs$party_family=="BJP"],na.rm=TRUE) -
                         mean(ip_rs$dim1[ip_rs$party_family=="INC"],na.rm=TRUE)) /
                     max(abs(mean(ip_rs$dim1[ip_rs$party_family=="BJP"],na.rm=TRUE)),
                         abs(mean(ip_rs$dim1[ip_rs$party_family=="INC"],na.rm=TRUE)))))
)

write_csv(summary_tbl, file.path(TABDIR, "compare_summary.csv"))
cat("  Saved compare_summary.csv\n")
#}

cat("\n[N2] All comparison figures saved.\n")
