# =============================================================================
# L1_rs_idealpoints.R
# Author: Piyush Zaware
# Updated: 2026-06-15
#
# Goal: Test whether Rajya Sabha members cluster by party in question
#       vocabulary -- the direct test of the J1 ideal points hypothesis.
#
#       J1 found NO party clustering in Lok Sabha starred questions:
#       constituency drives the vocabulary, not party.
#
#       Rajya Sabha members represent STATES (not individual constituencies).
#       If the J1 finding is driven by constituency variation, RS members
#       should show MORE party-line clustering.
#
# Inputs:
#   tmp/rajyasabha_raw.parquet     (anudit/rajyasabha-qa from HuggingFace)
#   input/rs_party_lookup.csv      (Wikipedia RS election data 2014-2024)
#
# Outputs:
#   output/figures/rs_ideal_2d.png
#   output/figures/rs_vs_ls_comparison.png
#   output/figures/rs_dim1_party.png
#   output/tables/rs_ideal_points.csv
#   output/tables/rs_party_centroids.csv
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

pkgs <- c("arrow","tidyverse","tidytext","Matrix","irlba","ggrepel","scales","ggridges")
to_inst <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_inst)) install.packages(to_inst)
suppressPackageStartupMessages(lapply(pkgs, library, character.only = TRUE))

FIGDIR  <- file.path(OUTDIR, "figures")
TABDIR  <- file.path(OUTDIR, "tables")
NAVY    <- "#0D1B2A"
BJP_COL <- "#FF9933"
INC_COL <- "#19AAED"

source(file.path(CODDIR, "._stop_words.R"))

MIN_Q         <- 5     # minimum starred questions per member
MIN_WORD_DOCS <- 5     # word must appear in >= 5 members' corpora

# =============================================================================
# SECTION 1: Load Rajya Sabha question data
# =============================================================================
#{
cat("[L1] Loading Rajya Sabha questions...\n")

rs_raw <- read_parquet(file.path(TMPDIR, "rajyasabha_clean.parquet")) %>%
  mutate(
    qtype   = str_to_upper(str_trim(qtype)),
    year    = as.integer(str_sub(as.character(adate), 1, 4)),
    english = str_replace_all(replace_na(english, ""), "\\x00", " ")
  ) %>%
  filter(qtype == "STARRED", year >= 2014, nchar(english) > 50)

cat("  Starred questions (2014+):", nrow(rs_raw), "\n")
cat("  Unique members:", n_distinct(rs_raw$name), "\n")
cat("  Year range:", min(rs_raw$year), "-", max(rs_raw$year), "\n")
#}

# =============================================================================
# SECTION 2: Name normalization and party matching
# =============================================================================
#{
cat("[L1] Matching RS members to parties...\n")

# Normalize names: remove honorifics, collapse spaces, uppercase
strip_hon_rs <- function(s) {
  s <- str_to_upper(str_squish(replace_na(s, "")))
  s <- str_remove_all(s, "\\b(SHRI|SHRIMATI|SMT\\.?|DR\\.?|PROF\\.?|MR\\.?|MRS\\.?|MS\\.?|SH\\.?|LATE)\\b")
  str_squish(s)
}

rs_raw <- rs_raw %>%
  mutate(name_norm = strip_hon_rs(name))

# Load party lookup
party_lookup <- read_csv(file.path(INPDIR, "rs_party_lookup.csv"),
                         show_col_types = FALSE) %>%
  mutate(
    name_norm = strip_hon_rs(name),
    party_family = case_when(
      party %in% c("BJP")                        ~ "BJP",
      party %in% c("INC")                        ~ "INC",
      party %in% c("Left")                       ~ "Left",
      party %in% c("TMC","AITC")                 ~ "TMC",
      party %in% c("SP")                         ~ "SP",
      party %in% c("BSP")                        ~ "BSP",
      party %in% c("JDU","JD(U)")                ~ "JDU",
      party %in% c("DMK")                        ~ "DMK",
      party %in% c("BJD")                        ~ "BJD",
      party %in% c("TDP")                        ~ "TDP",
      party %in% c("TRS","BRS")                  ~ "TRS",
      party %in% c("RJD")                        ~ "RJD",
      party %in% c("AIADMK")                     ~ "AIADMK",
      party %in% c("NCP")                        ~ "NCP",
      party %in% c("YSRCP","YSR")                ~ "YSRCP",
      party %in% c("AAP")                        ~ "AAP",
      TRUE                                        ~ "Other"
    )
  ) %>%
  arrange(desc(elected_year)) %>%
  distinct(name_norm, .keep_all = TRUE)

mp_party_rs <- setNames(party_lookup$party_family, party_lookup$name_norm)

# Fuzzy match: for each RS question, try exact match then token overlap
fuzzy_party <- function(name, lookup_names, lookup_parties, thresh = 0.5) {
  if (name %in% lookup_names) return(lookup_parties[name])
  tokens <- str_split(name, "\\s+")[[1]]
  best_score <- 0; best_party <- NA_character_
  for (i in seq_along(lookup_names)) {
    lt <- str_split(lookup_names[i], "\\s+")[[1]]
    score <- length(intersect(tokens, lt)) / max(length(union(tokens, lt)), 1)
    if (score > best_score) { best_score <- score; best_party <- lookup_parties[i] }
  }
  if (best_score >= thresh) best_party else NA_character_
}

lookup_names   <- names(mp_party_rs)
lookup_parties <- unname(mp_party_rs)

rs_raw <- rs_raw %>%
  mutate(
    party_family = mp_party_rs[name_norm],
    party_family = if_else(
      is.na(party_family),
      vapply(name_norm, fuzzy_party, character(1),
             lookup_names = lookup_names, lookup_parties = lookup_parties),
      party_family
    )
  )

matched <- mean(!is.na(rs_raw$party_family))
cat(sprintf("  Party match rate: %.1f%%\n", matched * 100))
cat("  Party breakdown:\n")
print(rs_raw %>% filter(!is.na(party_family)) %>% count(party_family) %>%
      arrange(-n) %>% as.data.frame())
#}

# =============================================================================
# SECTION 3: Build member-level TF-IDF matrix
# =============================================================================
#{
cat("[L1] Building TF-IDF matrix...\n")

starred_rs <- rs_raw %>% filter(!is.na(party_family))

# Members with enough questions
member_q_counts <- starred_rs %>%
  count(name_norm, name = "n_q") %>%
  filter(n_q >= MIN_Q)

tokens_rs <- starred_rs %>%
  semi_join(member_q_counts, by = "name_norm") %>%
  unnest_tokens(word, english) %>%
  filter(
    !word %in% COMBINED_STOP,
    str_detect(word, "^[a-z]{5,}$")
  )

word_doc_freq <- tokens_rs %>%
  distinct(name_norm, word) %>%
  count(word) %>%
  filter(n >= MIN_WORD_DOCS)

tokens_rs <- tokens_rs %>% filter(word %in% word_doc_freq$word)

cat(sprintf("  Vocabulary: %d words | Members: %d\n",
            nrow(word_doc_freq), n_distinct(tokens_rs$name_norm)))

tfidf_rs <- tokens_rs %>%
  count(name_norm, word) %>%
  bind_tf_idf(word, name_norm, n)
#}

# =============================================================================
# SECTION 4: Truncated SVD (same as J1 for comparability)
# =============================================================================
#{
cat("[L1] Running SVD...\n")

mp_idx   <- tfidf_rs %>% distinct(name_norm) %>% mutate(i = row_number())
word_idx <- tfidf_rs %>% distinct(word)      %>% mutate(j = row_number())

mat_df <- tfidf_rs %>%
  left_join(mp_idx,   by = "name_norm") %>%
  left_join(word_idx, by = "word")

M <- sparseMatrix(
  i    = mat_df$i,
  j    = mat_df$j,
  x    = mat_df$tf_idf,
  dims = c(nrow(mp_idx), nrow(word_idx)),
  dimnames = list(mp_idx$name_norm, word_idx$word)
)

cat(sprintf("  Matrix: %d members x %d words\n", nrow(M), ncol(M)))

# Column-center (same as J1: captures deviation from average, not raw frequency)
col_means  <- Matrix::colMeans(M)
M_centered <- M - matrix(col_means, nrow=nrow(M), ncol=ncol(M), byrow=TRUE)

svd_rs   <- irlba(M_centered, nv = 3)
var_expl <- svd_rs$d^2 / sum(svd_rs$d^2)

cat(sprintf("  Variance: D1=%.1f%%  D2=%.1f%%  D3=%.1f%%\n",
            var_expl[1]*100, var_expl[2]*100, var_expl[3]*100))

rs_coords <- tibble(
  name_norm = rownames(M),
  dim1 = svd_rs$u[,1] * svd_rs$d[1],
  dim2 = svd_rs$u[,2] * svd_rs$d[2]
)

# Attach party
mp_party_tbl_rs <- starred_rs %>%
  distinct(name_norm, party_family) %>%
  group_by(name_norm) %>% slice(1) %>% ungroup()

rs_ideal <- rs_coords %>%
  left_join(mp_party_tbl_rs, by = "name_norm") %>%
  mutate(party_family = replace_na(party_family, "Other"))

rs_centroids <- rs_ideal %>%
  group_by(party_family) %>%
  filter(n() >= 5) %>%
  summarise(dim1=mean(dim1), dim2=mean(dim2), n=n(),
            dim1_sd=sd(dim1), .groups="drop")

write_csv(rs_ideal,      file.path(TABDIR, "rs_ideal_points.csv"))
write_csv(rs_centroids,  file.path(TABDIR, "rs_party_centroids.csv"))

cat("\n=== RS Party centroids (Dim1) ===\n")
rs_centroids %>% arrange(dim1) %>%
  select(party_family, n, dim1, dim2) %>% print()
#}

# =============================================================================
# SECTION 5: Figure 1 -- RS 2D ideal point scatter
# =============================================================================
#{
cat("\n[L1] Figure 1: RS 2D scatter...\n")

party_colors <- c(
  BJP="FF9933", INC="#19AAED", Left="#CC0000", TMC="#2ECC71",
  SP="#8E44AD", BSP="#3498DB", JDU="#27AE60", DMK="#E67E22",
  BJD="#95A5A6", TDP="#DAA520", TRS="#16A085", RJD="#E74C3C",
  Other="#BDC3C7"
)
party_colors["BJP"] <- "#FF9933"

p_rs_2d <- ggplot(rs_ideal, aes(x=dim1, y=dim2)) +
  geom_point(aes(color=party_family), alpha=0.35, size=1.2) +
  geom_point(data=rs_centroids, aes(color=party_family),
             size=6, shape=18, alpha=0.95) +
  geom_label_repel(
    data=rs_centroids,
    aes(label=paste0(party_family,"\n(n=",n,")"), color=party_family),
    fontface="bold", size=3.2, box.padding=0.6,
    label.padding=unit(0.15,"lines"), max.overlaps=15, fill="white"
  ) +
  scale_color_manual(values=party_colors, na.value="#95A5A6") +
  labs(
    title    = "Rajya Sabha ideal points from question vocabulary",
    subtitle = sprintf(
      "Each point = one RS member. Diamonds = party centroids.\nDim 1: %.1f%% variance  |  Dim 2: %.1f%% variance",
      var_expl[1]*100, var_expl[2]*100
    ),
    x       = sprintf("Dimension 1 (%.1f%% variance)", var_expl[1]*100),
    y       = sprintf("Dimension 2 (%.1f%% variance)", var_expl[2]*100),
    color   = NULL,
    caption = "Starred questions, 2014-2025. Stop words and honorifics removed."
  ) +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold", colour=NAVY, size=14),
        plot.subtitle=element_text(colour="grey35", size=10),
        legend.position="none")

ggsave(file.path(FIGDIR,"rs_ideal_2d.png"), p_rs_2d, width=10, height=7, dpi=180)
file.copy(file.path(FIGDIR,"rs_ideal_2d.png"),
          file.path(dirname(OUTDIR),"docs","output","figures","rs_ideal_2d.png"),
          overwrite=TRUE)
cat("  Saved rs_ideal_2d.png\n")
#}

# =============================================================================
# SECTION 6: Figure 2 -- Dim 1 ridge plot by party (RS)
# =============================================================================
#{
cat("[L1] Figure 2: RS Dim 1 ridge plot...\n")

big_parties_rs <- rs_ideal %>% count(party_family) %>%
  filter(n>=5) %>% pull(party_family)

party_order_rs <- rs_ideal %>%
  filter(party_family %in% big_parties_rs) %>%
  group_by(party_family) %>%
  summarise(med=median(dim1),.groups="drop") %>%
  arrange(med) %>% pull(party_family)

p_rs_dim1 <- rs_ideal %>%
  filter(party_family %in% big_parties_rs) %>%
  mutate(party_family=factor(party_family, levels=party_order_rs)) %>%
  ggplot(aes(x=dim1, y=party_family, fill=party_family)) +
  geom_density_ridges(alpha=0.7, scale=0.9, rel_min_height=0.01) +
  scale_fill_manual(values=party_colors, na.value="#95A5A6") +
  labs(
    title    = "Rajya Sabha Dimension 1: does party separate here?",
    subtitle = "Distribution of RS member ideal points along primary SVD dimension",
    x        = sprintf("Dimension 1 (%.1f%% variance)", var_expl[1]*100),
    y        = NULL, fill=NULL,
    caption  = "Starred questions 2014-2025. Compare with Lok Sabha figure from J1."
  ) +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold", colour=NAVY, size=14),
        legend.position="none")

ggsave(file.path(FIGDIR,"rs_dim1_party.png"), p_rs_dim1, width=9, height=6, dpi=180)
file.copy(file.path(FIGDIR,"rs_dim1_party.png"),
          file.path(dirname(OUTDIR),"docs","output","figures","rs_dim1_party.png"),
          overwrite=TRUE)
cat("  Saved rs_dim1_party.png\n")
#}

# =============================================================================
# SECTION 7: Figure 3 -- RS vs LS centroid separation comparison
# =============================================================================
#{
cat("[L1] Figure 3: RS vs LS centroid comparison...\n")

# Load LS centroids from J1
ls_centroids_path <- file.path(TABDIR, "ideal_points.csv")
if (file.exists(ls_centroids_path)) {
  ls_ideal <- read_csv(ls_centroids_path, show_col_types=FALSE)
  ls_centroids <- ls_ideal %>%
    group_by(party_family) %>%
    filter(n() >= 5) %>%
    summarise(dim1=mean(dim1), dim2=mean(dim2), n=n(), .groups="drop") %>%
    mutate(house="Lok Sabha")

  rs_centroids_comp <- rs_centroids %>%
    filter(party_family %in% ls_centroids$party_family) %>%
    select(party_family, dim1, dim2, n) %>%
    mutate(house="Rajya Sabha")

  # Normalise both to [-1, 1] within each house for comparability
  norm_dim1 <- function(x) (x - mean(x)) / (max(abs(x - mean(x))) + 1e-10)

  comp <- bind_rows(
    ls_centroids %>% mutate(dim1_norm = norm_dim1(dim1)),
    rs_centroids_comp %>% mutate(dim1_norm = norm_dim1(dim1))
  ) %>%
    filter(party_family %in% c("BJP","INC","Left","TMC","SP","JDU","DMK","BJD","TDP"))

  # Spread between BJP and INC on Dim1 -- key diagnostic
  bjp_inc <- comp %>% filter(party_family %in% c("BJP","INC")) %>%
    group_by(house) %>%
    summarise(spread=diff(range(dim1_norm)), .groups="drop")
  cat("\n=== BJP-INC Dim1 spread (normalised) ===\n")
  print(bjp_inc)

  p_comp <- ggplot(comp, aes(x=dim1_norm, y=fct_reorder(party_family, dim1_norm),
                               color=house, shape=house)) +
    geom_point(aes(size=n), alpha=0.85) +
    geom_line(aes(group=party_family), color="grey70", linewidth=0.5) +
    scale_color_manual(values=c("Lok Sabha"="#2C3E50","Rajya Sabha"="#E74C3C"),
                       name=NULL) +
    scale_shape_manual(values=c("Lok Sabha"=16,"Rajya Sabha"=17), name=NULL) +
    scale_size_continuous(range=c(3,9), guide="none") +
    geom_vline(xintercept=0, linetype="dashed", color="grey60") +
    labs(
      title    = "Do Rajya Sabha parties separate more than Lok Sabha parties?",
      subtitle = paste0(
        "Normalised party centroids on primary SVD dimension.\n",
        "Wider spread = more party-based vocabulary differentiation."
      ),
      x       = "Normalised Dimension 1 (within-house)",
      y       = NULL,
      caption = "Lines connect the same party across the two houses. Larger points = more members."
    ) +
    theme_minimal(base_size=12) +
    theme(
      plot.title    = element_text(face="bold", colour=NAVY, size=13),
      plot.subtitle = element_text(colour="grey35", size=10),
      legend.position = "bottom"
    )

  ggsave(file.path(FIGDIR,"rs_vs_ls_comparison.png"),
         p_comp, width=10, height=7, dpi=180)
  file.copy(file.path(FIGDIR,"rs_vs_ls_comparison.png"),
            file.path(dirname(OUTDIR),"docs","output","figures","rs_vs_ls_comparison.png"),
            overwrite=TRUE)
  cat("  Saved rs_vs_ls_comparison.png\n")
} else {
  cat("  J1 ideal_points.csv not found -- skipping LS comparison figure\n")
}
#}

# =============================================================================
# SECTION 8: Key statistics for article
# =============================================================================
#{
cat("\n========== KEY STATS FOR ARTICLE ==========\n")
cat("RS members in analysis:", nrow(rs_ideal), "\n")
cat("Questions analysed:", nrow(starred_rs %>% semi_join(member_q_counts, by='name_norm')), "\n\n")

# Within-party variance vs between-party variance
big_parties_stats <- rs_ideal %>% filter(party_family %in% big_parties_rs)
between_var <- var(rs_centroids %>% filter(party_family %in% big_parties_rs) %>% pull(dim1))
within_var  <- big_parties_stats %>%
  group_by(party_family) %>%
  summarise(v=var(dim1),.groups="drop") %>%
  pull(v) %>% mean()

cat(sprintf("Between-party variance (Dim1): %.6f\n", between_var))
cat(sprintf("Within-party variance (Dim1):  %.6f\n", within_var))
cat(sprintf("Ratio (between/within):         %.3f\n", between_var/within_var))
cat("\nFor comparison, J1 Lok Sabha BJP centroid: 0.000894, INC: 0.000940\n")
cat("(Near-zero between-party variance)\n")

cat("\nRS party centroids sorted by Dim1:\n")
rs_centroids %>% arrange(dim1) %>%
  select(party_family, n, dim1, dim2) %>%
  mutate(across(where(is.numeric), ~round(.x, 5))) %>%
  print()
#}

cat("\n[L1] Done.\n")
