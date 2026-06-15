# =============================================================================
# G2b_clean_figures.R
# Author: Piyush Zaware
# Updated: 2026-06-14
#
# Regenerates the two article figures from G2 with PDF artifact fragments
# explicitly blocked. Requires G2 to have already run (uses its TF-IDF objects,
# or recomputes them if not in environment).
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

pkgs <- c("arrow","tidyverse","tidytext","scales")
suppressPackageStartupMessages(lapply(pkgs, library, character.only = TRUE))

FIGDIR <- file.path(OUTDIR, "figures")
NAVY   <- "#0D1B2A"
BJP_COL <- "#FF9933"
INC_COL <- "#19AAED"

source(file.path(CODDIR, "._stop_words.R"))

# PDF hyphenation fragments and known OCR artifacts to block explicitly.
# These slip through the cross-corpus filter because they appear in both
# manifesto PDFs and parliamentary question PDFs as extraction noise.
PDF_ARTIFACTS <- c(
  # PDF hyphenation fragments
  "nancial", "ation",  "tion",   "asstt",  "hvics",
  "chinnaya","commi",  "suppo",  "icipation","ective",
  "heastern","airpo",  "signi",  "transpo", "electri",
  "cation",  "unities","rancial","opment",  "tment",
  "overn",   "ealth",  "ructure","dustry",  "iness",
  "sation",  "ficial", "roduct", "artment", "ource",
  "ntment",  "liance", "ssion",  "lement",  "ission",
  # Specific confirmed artifacts from visual inspection
  "ciaries", "benefi", "bbssl",  "makthappa",
  "shall",   "utilizing", "compatible", "combine",
  "bbsl",    "focussed"
)

# =============================================================================
# Recompute TF-IDF matrices (fast: reuses stop word list + parquet files)
# =============================================================================
#{
cat("[G2b] Building TF-IDF matrices with artifact filter...\n")

manifesto_raw <- read_csv(file.path(OUTDIR, "tables", "manifesto_text.csv"),
                           show_col_types = FALSE) %>%
  filter(lok_no >= 16, n_chars > 5000) %>%
  mutate(party_q = recode(party,
    "BJP"="BJP","INC"="INC","AITC"="AITC","NCP"="NCP",
    "AIADMK"="AIADMK","CPI-M"="CPI(M)","DMK"="DMK",.default=party))

parquet_files <- list.files(TMPDIR, pattern="train-.*\\.parquet$", full.names=TRUE)
raw <- map_dfr(parquet_files, function(f)
  read_parquet(f, col_select=c("lok_no","type","members","question_text")))

strip_hon <- function(s) {
  s <- str_to_upper(str_squish(s))
  str_remove_all(s,"\\b(SHRIMATI|SMT\\.?|KUMARI|MRS\\.?|MS\\.?|DR\\.?|PROF\\.?|SH\\.?|SHRI\\.?)\\b")
}
norm_fl <- function(s) {
  parts <- str_split(str_squish(s),"\\s+")[[1]]
  if (length(parts) <= 2) return(s)
  paste(parts[1], parts[length(parts)])
}

lookup <- read_csv(file.path(INPDIR,"mp_party_lookup.csv"), show_col_types=FALSE) %>%
  mutate(mp_norm = vapply(vapply(str_to_upper(str_squish(mp_name)),
                                  strip_hon, character(1)), norm_fl, character(1))) %>%
  arrange(desc(lok_no)) %>% distinct(mp_norm, .keep_all=TRUE)
mp_party <- setNames(lookup$party_family, lookup$mp_norm)

starred <- raw %>%
  filter(type=="STARRED", lok_no>=16) %>%
  mutate(
    primary_raw  = map_chr(members, function(x)
      tryCatch(str_squish(as.character(list(x)[[1]])[1]), error=function(e) NA_character_)),
    primary_norm = vapply(vapply(replace_na(primary_raw,""), strip_hon, character(1)),
                           norm_fl, character(1)),
    party_family = mp_party[primary_norm]
  ) %>%
  filter(!is.na(party_family), !is.na(question_text))

# Question word counts with artifact filter
q_words <- starred %>%
  unnest_tokens(word, question_text) %>%
  filter(!word %in% COMBINED_STOP,
         !word %in% PDF_ARTIFACTS,
         str_detect(word,"^[a-z]+$"), nchar(word) >= 5) %>%
  count(party_family, lok_no, word, name="n_q")

q_valid_words <- q_words %>% distinct(word) %>% pull(word)

# Manifesto word counts with artifact filter
man_words <- manifesto_raw %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% COMBINED_STOP,
         !word %in% PDF_ARTIFACTS,
         str_detect(word,"^[a-z]+$"), nchar(word) >= 5,
         word %in% q_valid_words) %>%
  count(party_q, lok_no, word, name="n_man")

# TF-IDF
man_tfidf <- man_words %>%
  mutate(doc_id = paste(party_q, lok_no, sep="_")) %>%
  bind_tf_idf(word, doc_id, n_man)

q_tfidf <- q_words %>%
  mutate(doc_id = paste(party_family, lok_no, sep="_")) %>%
  bind_tf_idf(word, doc_id, n_q)

# Party-level averages
bjp_man <- man_tfidf %>% filter(party_q=="BJP") %>%
  group_by(word) %>% summarise(man_tfidf=mean(tf_idf), .groups="drop")

bjp_q <- q_tfidf %>% filter(party_family=="BJP") %>%
  group_by(word) %>% summarise(q_tfidf=mean(tf_idf), .groups="drop")

inc_q <- q_tfidf %>% filter(party_family=="INC") %>%
  group_by(word) %>% summarise(inc_q_tfidf=mean(tf_idf), .groups="drop")

cat("  Done. Vocabulary sizes:\n")
cat("    BJP manifesto:", nrow(bjp_man), "words\n")
cat("    BJP questions:", nrow(bjp_q),   "words\n")
cat("    INC questions:", nrow(inc_q),   "words\n")
#}

# =============================================================================
# FIGURE 1: BJP promise gap (clean)
# =============================================================================
#{
cat("[G2b] Plotting clean BJP promise gap...\n")

bjp_gap <- full_join(bjp_man, bjp_q, by="word") %>%
  replace_na(list(man_tfidf=0, q_tfidf=0)) %>%
  mutate(
    gap       = man_tfidf - q_tfidf,
    direction = case_when(
      gap >  0.0002 ~ "Promised but not questioned in Parliament",
      gap < -0.0002 ~ "Questioned in Parliament beyond manifesto",
      TRUE          ~ "Aligned"
    )
  ) %>%
  filter(man_tfidf + q_tfidf > 0.0001)

top_gap <- bind_rows(
  bjp_gap %>% filter(direction=="Promised but not questioned in Parliament") %>%
    slice_max(gap, n=18),
  bjp_gap %>% filter(direction=="Questioned in Parliament beyond manifesto") %>%
    slice_min(gap, n=12)
) %>%
  mutate(
    word  = str_to_title(word),
    word  = fct_reorder(word, gap)
  )

p_gap <- ggplot(top_gap, aes(x=gap, y=word, fill=direction)) +
  geom_col(width=0.72) +
  geom_vline(xintercept=0, linewidth=0.5, colour="grey40") +
  scale_fill_manual(
    values = c(
      "Promised but not questioned in Parliament"  = BJP_COL,
      "Questioned in Parliament beyond manifesto"  = NAVY
    ),
    name = NULL
  ) +
  scale_x_continuous(labels=scales::label_number(accuracy=0.001)) +
  labs(
    title    = "BJP: what they promised vs what they questioned",
    subtitle = paste0(
      "Orange = prominent in BJP manifestos (2014-2024) but largely absent from BJP MPs' starred questions.\n",
      "Navy = topics BJP MPs questioned extensively that barely appear in any of the manifestos."
    ),
    x = "TF-IDF gap (manifesto weight minus question weight)",
    y = NULL,
    caption = "Averaged across 16th, 17th and 18th Lok Sabha. PDF artifacts and parliamentary boilerplate removed."
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face="bold", colour=NAVY, size=14),
    plot.subtitle    = element_text(colour="grey35", size=10),
    plot.caption     = element_text(colour="grey55", size=8),
    legend.position  = "bottom",
    legend.text      = element_text(size=10),
    panel.grid.major.y = element_blank(),
    axis.text.y      = element_text(size=11)
  )

ggsave(file.path(FIGDIR, "manifesto_bjp_promise_gap.png"),
       p_gap, width=10, height=9, dpi=180)
# Copy to docs for GitHub Pages
file.copy(file.path(FIGDIR,"manifesto_bjp_promise_gap.png"),
          file.path(dirname(OUTDIR),"docs","output","figures","manifesto_bjp_promise_gap.png"),
          overwrite=TRUE)
cat("  Saved manifesto_bjp_promise_gap.png\n")
#}

# =============================================================================
# FIGURE 2: Opposition accountability (clean)
# =============================================================================
#{
cat("[G2b] Plotting clean opposition accountability...\n")

inc_accountability <- full_join(bjp_man, inc_q, by="word") %>%
  replace_na(list(man_tfidf=0, inc_q_tfidf=0)) %>%
  mutate(
    in_bjp_man = man_tfidf > 0.0001,
    in_inc_q   = inc_q_tfidf > 0.0001
  )

acc_long <- bind_rows(
  inc_accountability %>%
    filter(in_bjp_man) %>%
    slice_max(inc_q_tfidf, n=20) %>%
    mutate(panel="BJP promise words INC MPs DO question"),
  inc_accountability %>%
    filter(in_bjp_man, !in_inc_q) %>%
    slice_max(man_tfidf, n=18) %>%
    mutate(panel="BJP promise words INC MPs DON'T question")
) %>%
  mutate(
    word  = str_to_title(word),
    word  = fct_reorder(word, inc_q_tfidf + man_tfidf),
    panel = factor(panel, levels=c(
      "BJP promise words INC MPs DO question",
      "BJP promise words INC MPs DON'T question"
    ))
  )

panel_labels <- c(
  "BJP promise words INC MPs DO question"      = "BJP promises INC DOES hold them to",
  "BJP promise words INC MPs DON'T question"   = "BJP promises INC largely ignores"
)

p_acc <- acc_long %>%
  ggplot(aes(x=inc_q_tfidf + man_tfidf,
             y=word,
             fill=panel)) +
  geom_col(width=0.72) +
  facet_wrap(~panel, scales="free_y", ncol=2, labeller=as_labeller(panel_labels)) +
  scale_fill_manual(
    values=c(
      "BJP promise words INC MPs DO question"      = INC_COL,
      "BJP promise words INC MPs DON'T question"   = "grey55"
    ),
    guide="none"
  ) +
  scale_x_continuous(labels=scales::label_number(accuracy=0.001)) +
  labs(
    title    = "Does the opposition hold BJP to its own promises?",
    subtitle = paste0(
      "Left: BJP manifesto words that Congress MPs actively raise in starred questions.\n",
      "Right: BJP promises that Congress largely lets pass without parliamentary scrutiny."
    ),
    x = "Combined TF-IDF weight",
    y = NULL,
    caption = "INC questions from 16th, 17th and 18th Lok Sabha. BJP manifestos 2014-2024. Artifacts removed."
  ) +
  theme_minimal(base_size=12) +
  theme(
    plot.title       = element_text(face="bold", colour=NAVY, size=14),
    plot.subtitle    = element_text(colour="grey35", size=10),
    plot.caption     = element_text(colour="grey55", size=8),
    strip.text       = element_text(face="bold", colour=NAVY, size=11),
    panel.grid.major.y = element_blank(),
    axis.text.y      = element_text(size=11)
  )

ggsave(file.path(FIGDIR, "manifesto_opposition_accountability.png"),
       p_acc, width=13, height=8, dpi=180)
file.copy(file.path(FIGDIR,"manifesto_opposition_accountability.png"),
          file.path(dirname(OUTDIR),"docs","output","figures","manifesto_opposition_accountability.png"),
          overwrite=TRUE)
cat("  Saved manifesto_opposition_accountability.png\n")
#}

cat("[G2b] Done.\n")
