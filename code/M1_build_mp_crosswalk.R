# =============================================================================
# M1_build_mp_crosswalk.R
# Author: Piyush Zaware
# Updated: 2026-06-15
#
# Goal: Build a robust raw_name -> party_family crosswalk for every MP name
#       that appears in the starred questions parquet files.
#
#       The current norm_fl() approach fails on four patterns:
#         1. Dr./Prof./Adv. prefixes -- period breaks regex word boundary
#         2. Word order differences (lookup: "RAM MOHAN NAIDU KINJARAPU",
#            questions: "KINJARAPU RAM MOHAN NAIDU")
#         3. norm_fl keeps wrong token (K. ASHOK KUMAR -> K. KUMAR not ASHOK KUMAR)
#         4. Shri/Dr prefix leaves leading whitespace
#
#       Fix: generate FIVE keys per MP name, try them in order.
#
# Inputs:
#   input/mp_party_lookup.csv
#   tmp/train-*.parquet
#
# Outputs:
#   input/mp_name_crosswalk.csv  (raw_name, party_family, lok_no, method)
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  if (!requireNamespace("stringdist", quietly = TRUE)) install.packages("stringdist")
  library(stringdist)
})

# =============================================================================
# SECTION 1: Improved name cleaning functions
# =============================================================================
#{

# Step 1: remove everything that isn't part of the real name
clean_name <- function(s) {
  s <- str_to_upper(str_squish(replace_na(s, "")))
  # Remove parenthesized prefixes like (ADV.) or (RETD.) or nicknames like (NANA)
  s <- str_remove_all(s, "\\(.*?\\)")
  # Normalize "A.B." compound initials to "A. B." so token splitting aligns
  s <- str_replace_all(s, "([A-Z])\\.([A-Z])", "\\1. \\2")
  # Remove leading titles that end with a period -- these break \\b matching
  s <- str_remove(s, "^(ADV|PROF|COL|BRIG|GEN|DR|MR|MRS|MS|ER|CAPT|MAJOR|LT|PR)\\.\\s*")
  # Remove remaining honorifics with word boundaries
  s <- str_remove_all(s,
    "\\b(SHRIMATI|SMT|KUMARI|SH|SHRI|MRS|MR|MS|DR|PROF|ADV|COL|TEACHER)\\b\\.?")
  # Remove trailing single-letter initials: "PARASURAMAN K." or "SELVAM G"
  s <- str_remove(s, "\\s+[A-Z]\\.?$")
  # Remove leading/trailing punctuation and spaces
  s <- str_remove_all(s, "^[^A-Z]+|[^A-Z]+$")
  str_squish(s)
}

# Step 2: generate multiple matching keys from a cleaned name
make_keys <- function(s) {
  s <- clean_name(s)
  tokens <- str_split(s, "\\s+")[[1]]
  # Drop standalone initials like "B." "S.P." "K."
  real_tokens <- tokens[!str_detect(tokens, "^[A-Z]{1,2}\\.?$")]
  if (length(real_tokens) == 0) real_tokens <- tokens

  keys <- character(0)

  # Key 1: full cleaned name
  keys[1] <- s

  # Key 2: first + last of real tokens
  if (length(real_tokens) >= 2) {
    keys[2] <- paste(real_tokens[1], real_tokens[length(real_tokens)])
  } else {
    keys[2] <- paste(real_tokens, collapse = " ")
  }

  # Key 3: sorted real tokens (catches word order reversal)
  keys[3] <- paste(sort(real_tokens), collapse = " ")

  # Key 4: last name only (single-token fallback -- only use if unique in lookup)
  keys[4] <- real_tokens[length(real_tokens)]

  # Key 5: first real token only
  keys[5] <- real_tokens[1]

  unique(keys[nchar(keys) > 2])
}

cat("[M1] Name cleaning functions defined\n")
#}

# =============================================================================
# SECTION 2: Build lookup key table with multiple keys per MP
# =============================================================================
#{
cat("[M1] Building multi-key lookup from mp_party_lookup.csv...\n")

lookup <- read_csv(file.path(INPDIR, "mp_party_lookup.csv"), show_col_types = FALSE) %>%
  mutate(
    party_family = case_when(
      str_detect(party, "Bharatiya Janata")     ~ "BJP",
      str_detect(party, "Indian National Cong") ~ "INC",
      str_detect(party, "Communist|CPI|CPM")    ~ "Left",
      str_detect(party, "Trinamool|AITC")       ~ "TMC",
      str_detect(party, "Telugu Desam")         ~ "TDP",
      str_detect(party, "Samajwadi")            ~ "SP",
      str_detect(party, "Bahujan Samaj")        ~ "BSP",
      str_detect(party, "Janata Dal.*United|JD.*U") ~ "JDU",
      str_detect(party, "Dravida Munnetra")     ~ "DMK",
      str_detect(party, "AIADMK|All India Anna")~ "AIADMK",
      str_detect(party, "Shiv Sena")            ~ "Shiv Sena",
      str_detect(party, "Rashtriya Janata")     ~ "RJD",
      str_detect(party, "Nationalist Congress") ~ "NCP",
      str_detect(party, "YSR|Yuvajana")         ~ "YSRCP",
      str_detect(party, "Aam Aadmi")            ~ "AAP",
      str_detect(party, "Independent")          ~ "Independent",
      TRUE                                      ~ "Regional"
    )
  )

# For each MP in each lok_no, generate all keys
lookup_keys <- lookup %>%
  rowwise() %>%
  mutate(keys = list(make_keys(mp_name))) %>%
  ungroup() %>%
  unnest(keys) %>%
  rename(lookup_key = keys) %>%
  filter(nchar(lookup_key) > 2) %>%
  # If a key maps to multiple parties within same lok_no, drop it (ambiguous)
  group_by(lookup_key, lok_no) %>%
  mutate(n_parties = n_distinct(party_family)) %>%
  filter(n_parties == 1) %>%
  ungroup() %>%
  distinct(lookup_key, lok_no, party_family, mp_name)

# Also build a lok_no-agnostic version (most recent entry wins)
lookup_keys_any <- lookup_keys %>%
  arrange(desc(lok_no)) %>%
  distinct(lookup_key, .keep_all = TRUE)

cat(sprintf("  %d unique lookup keys from %d MPs\n",
            n_distinct(lookup_keys$lookup_key), n_distinct(lookup$mp_name)))
#}

# =============================================================================
# SECTION 3: Extract all unique MP names from question parquets
# =============================================================================
#{
cat("[M1] Extracting unique MP names from parquet files...\n")

parquet_files <- list.files(TMPDIR, pattern = "train-.*\\.parquet$", full.names = TRUE)
raw <- map_dfr(parquet_files, function(f)
  read_parquet(f, col_select = c("lok_no", "type", "members"))) %>%
  filter(type == "STARRED", lok_no >= 16L) %>%
  mutate(
    raw_name = map_chr(members, function(x)
      tryCatch(str_squish(as.character(list(x)[[1]])[1]),
               error = function(e) NA_character_))
  ) %>%
  filter(!is.na(raw_name), nchar(raw_name) > 2)

# Unique raw_name + lok_no combinations to match
to_match <- raw %>%
  distinct(raw_name, lok_no) %>%
  arrange(lok_no, raw_name)

cat(sprintf("  %d unique (name, lok_no) combinations to match\n", nrow(to_match)))
#}

# =============================================================================
# SECTION 4: Multi-pass matching
# =============================================================================
#{
cat("[M1] Running multi-pass matching...\n")

# Build fast lookup maps
lk_map    <- split(lookup_keys, lookup_keys$lok_no)
lk_map_any <- setNames(lookup_keys_any$party_family, lookup_keys_any$lookup_key)
lk_mp_any  <- setNames(lookup_keys_any$mp_name,      lookup_keys_any$lookup_key)

match_one <- function(raw_name, lok_no) {
  keys <- make_keys(raw_name)

  # Pass 1: exact key match within same lok_no
  lk <- lk_map[[as.character(lok_no)]]
  if (!is.null(lk)) {
    lk_map_lok <- setNames(lk$party_family, lk$lookup_key)
    for (k in keys) {
      if (k %in% names(lk_map_lok)) return(list(party = lk_map_lok[k], method = "exact_lok"))
    }
  }

  # Pass 2: exact key match across any lok_no
  for (k in keys) {
    if (k %in% names(lk_map_any)) return(list(party = lk_map_any[k], method = "exact_any"))
  }

  # Pass 3: token overlap -- find lookup entry where >=2 real tokens match
  clean  <- clean_name(raw_name)
  tokens <- str_split(clean, "\\s+")[[1]]
  tokens <- tokens[!str_detect(tokens, "^[A-Z]{1,2}\\.?$") & nchar(tokens) > 2]

  if (length(tokens) >= 2) {
    candidates <- lookup_keys_any %>%
      filter(str_detect(lookup_key, tokens[1]) | str_detect(lookup_key, tokens[length(tokens)]))
    for (i in seq_len(nrow(candidates))) {
      cand_tokens <- str_split(candidates$lookup_key[i], "\\s+")[[1]]
      n_overlap   <- sum(tokens %in% cand_tokens)
      if (n_overlap >= 2) {
        return(list(party = candidates$party_family[i], method = "token_overlap"))
      }
    }
  }

  # Pass 4: Jaro-Winkler fuzzy match on cleaned full name (>= 0.92 threshold)
  # Restrict to unambiguous keys (length >= 8) to avoid spurious single-token matches
  jw_keys   <- lookup_keys_any$lookup_key[nchar(lookup_keys_any$lookup_key) >= 8]
  jw_scores <- stringsim(clean, jw_keys, method = "jw", p = 0.1)
  best_idx  <- which.max(jw_scores)
  if (length(best_idx) > 0 && jw_scores[best_idx] >= 0.92) {
    best_key   <- jw_keys[best_idx]
    best_party <- lookup_keys_any$party_family[lookup_keys_any$lookup_key == best_key][1]
    return(list(party = best_party, method = "fuzzy_jw"))
  }

  list(party = NA_character_, method = "miss")
}

# Apply to all unique name × lok_no pairs
cat("  Matching", nrow(to_match), "combinations...\n")
results <- vector("list", nrow(to_match))
for (i in seq_len(nrow(to_match))) {
  results[[i]] <- match_one(to_match$raw_name[i], to_match$lok_no[i])
}

crosswalk <- to_match %>%
  mutate(
    party_family = map_chr(results, "party"),
    method       = map_chr(results, "method")
  )

# Summary
cat("\n[M1] Match results:\n")
crosswalk %>% count(method) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  print()

# Join back to raw questions to compute question-level match rate
raw_matched <- raw %>%
  left_join(crosswalk, by = c("raw_name", "lok_no"))

cat(sprintf("\nQuestion-level match rate: %.1f%%  (%d / %d)\n",
            100 * mean(!is.na(raw_matched$party_family)),
            sum(!is.na(raw_matched$party_family)),
            nrow(raw_matched)))

cat("\nBy Lok Sabha:\n")
raw_matched %>%
  group_by(lok_no) %>%
  summarise(
    total    = n(),
    matched  = sum(!is.na(party_family)),
    pct      = round(100 * matched / total, 1),
    .groups  = "drop"
  ) %>% print()
#}

# =============================================================================
# SECTION 5: Save crosswalk
# =============================================================================
#{
out_path <- file.path(INPDIR, "mp_name_crosswalk.csv")
crosswalk %>%
  filter(!is.na(party_family)) %>%
  write_csv(out_path)

cat(sprintf("\n[M1] Saved %d matched entries to %s\n",
            sum(!is.na(crosswalk$party_family)), out_path))

# Sample of still-unmatched names for manual inspection
cat("\nTop 30 unmatched names (by question frequency):\n")
raw_matched %>%
  filter(is.na(party_family)) %>%
  count(raw_name, sort = TRUE) %>%
  head(30) %>%
  print(n = 30)
#}

cat("\n[M1] Done.\n")
