# =============================================================================
# M2_build_rs_crosswalk.R
# Author: Piyush Zaware
# Updated: 2026-06-15
#
# Goal: Build a robust raw_name -> party_family crosswalk for Rajya Sabha
#       member names in the RS starred questions parquet.
#
# Key challenges vs LS crosswalk:
#   1. RS members serve overlapping 6-year terms -- same name can appear under
#      two different parties at different times (e.g., C. M. Ramesh TDP->BJP)
#   2. RS lookup covers only 287 of 476 unique member names (the rest are
#      genuinely absent from the Wikipedia source)
#   3. Many RS names are single-word or initial-heavy, so token overlap needs
#      to fire on fewer tokens
#
# Strategy: match on (name_key, year_window) not just name_key, so party
#           switches are resolved correctly. Fuzzy threshold relaxed to 0.88.
#
# Inputs:
#   input/rs_party_lookup.csv
#   tmp/rajyasabha_clean.parquet
#
# Outputs:
#   input/rs_name_crosswalk.csv  (raw_name, party_family, matched_rs_name, method)
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

RS_TERM_YEARS <- 6  # approximate RS term length

# =============================================================================
# SECTION 1: Name cleaning functions (same as M1)
# =============================================================================
#{
clean_name <- function(s) {
  s <- str_to_upper(str_squish(replace_na(s, "")))
  s <- str_remove_all(s, "\\(.*?\\)")
  s <- str_replace_all(s, "([A-Z])\\.([A-Z])", "\\1. \\2")
  s <- str_remove(s, "^(ADV|PROF|COL|BRIG|GEN|DR|MR|MRS|MS|ER|CAPT|MAJOR|LT|PR|LATE)\\.\\s*")
  s <- str_remove_all(s,
    "\\b(SHRIMATI|SMT|KUMARI|SH|SHRI|MRS|MR|MS|DR|PROF|ADV|COL|LATE)\\b\\.?")
  s <- str_remove(s, "\\s+[A-Z]\\.?$")
  # Remove apostrophe-based patterns (O' Brien -> O BRIEN)
  s <- str_replace_all(s, "'", "")
  s <- str_remove_all(s, "^[^A-Z]+|[^A-Z]+$")
  str_squish(s)
}

make_keys <- function(s) {
  s   <- clean_name(s)
  tks <- str_split(s, "\\s+")[[1]]
  # For RS, even names with 1 real token (like VAIKO) are valid
  real_tks <- tks[!str_detect(tks, "^[A-Z]{1,2}\\.?$")]
  if (length(real_tks) == 0) real_tks <- tks

  keys <- character(0)
  keys[1] <- s
  if (length(real_tks) >= 2) {
    keys[2] <- paste(real_tks[1], real_tks[length(real_tks)])
    keys[3] <- paste(sort(real_tks), collapse = " ")
  } else {
    keys[2] <- paste(real_tks, collapse = " ")
    keys[3] <- paste(real_tks, collapse = " ")
  }
  keys[4] <- real_tks[length(real_tks)]
  keys[5] <- real_tks[1]

  unique(keys[nchar(keys) > 1])
}

cat("[M2] Name cleaning functions defined\n")
#}

# =============================================================================
# SECTION 2: Build year-aware lookup key table
# =============================================================================
#{
cat("[M2] Building year-aware lookup from rs_party_lookup.csv + rs_supplement_lookup.csv...\n")

rs_lookup_main <- read_csv(file.path(INPDIR, "rs_party_lookup.csv"),
                           show_col_types = FALSE)
rs_lookup_supp <- read_csv(file.path(INPDIR, "rs_supplement_lookup.csv"),
                           show_col_types = FALSE)

rs_lookup <- bind_rows(rs_lookup_main, rs_lookup_supp) %>%
  distinct(name, elected_year, .keep_all = TRUE) %>%
  mutate(
    party_family = case_when(
      # rs_party_lookup.csv already stores some values as normalized families
      party %in% c("BJP")                    ~ "BJP",
      party %in% c("INC")                    ~ "INC",
      party %in% c("Left")                   ~ "Left",
      str_detect(party, "Communist|CPI|CPM") ~ "Left",
      party %in% c("TMC","AITC")             ~ "TMC",
      party %in% c("SP")                     ~ "SP",
      party %in% c("BSP")                    ~ "BSP",
      party %in% c("JDU","JD(U)")            ~ "JDU",
      party %in% c("DMK")                    ~ "DMK",
      party %in% c("BJD")                    ~ "BJD",
      party %in% c("TDP")                    ~ "TDP",
      party %in% c("TRS","BRS")              ~ "TRS",
      party %in% c("RJD")                    ~ "RJD",
      party %in% c("AIADMK")                ~ "AIADMK",
      party %in% c("NCP")                    ~ "NCP",
      party %in% c("YSRCP","YSR")           ~ "YSRCP",
      party %in% c("AAP")                    ~ "AAP",
      TRUE                                    ~ "Other"
    ),
    term_start = elected_year,
    term_end   = elected_year + RS_TERM_YEARS
  )

# Generate all keys per lookup entry (with year window)
lookup_keys_raw <- rs_lookup %>%
  rowwise() %>%
  mutate(keys = list(make_keys(name))) %>%
  ungroup() %>%
  unnest(keys) %>%
  rename(lookup_key = keys) %>%
  filter(nchar(lookup_key) > 1) %>%
  select(lookup_key, name, party_family, term_start, term_end)

cat(sprintf("  %d unique lookup keys from %d RS members\n",
            n_distinct(lookup_keys_raw$lookup_key), n_distinct(rs_lookup$name)))
#}

# =============================================================================
# SECTION 3: Extract unique (raw_name, year) combinations from RS parquet
# =============================================================================
#{
cat("[M2] Extracting unique (member, year) combinations...\n")

rs_raw <- read_parquet(file.path(TMPDIR, "rajyasabha_clean.parquet")) %>%
  mutate(
    qtype = str_to_upper(str_trim(qtype)),
    year  = as.integer(str_sub(as.character(adate), 1, 4))
  ) %>%
  filter(qtype == "STARRED", year >= 2014) %>%
  mutate(raw_name = str_squish(replace_na(as.character(name), ""))) %>%
  filter(nchar(raw_name) > 1)

# We match on raw_name alone (since a given member typically serves 6 years,
# their party assignment within that window is stable; we use year only for
# the 3 known party-switchers)
year_by_member <- rs_raw %>%
  group_by(raw_name) %>%
  summarise(question_year = median(year, na.rm = TRUE), .groups = "drop")

to_match <- rs_raw %>%
  distinct(raw_name) %>%
  left_join(year_by_member, by = "raw_name") %>%
  arrange(raw_name)

cat(sprintf("  %d unique member names to match\n", nrow(to_match)))
#}

# =============================================================================
# SECTION 4: Multi-pass matching with year-window disambiguation
# =============================================================================
#{
cat("[M2] Running multi-pass matching...\n")

match_one_rs <- function(raw_name, qyear) {
  keys  <- make_keys(raw_name)
  clean <- clean_name(raw_name)

  # Helper: from candidates keyed by lookup_key, pick the one whose term
  # covers qyear; if none, pick the closest
  best_match <- function(candidates) {
    if (nrow(candidates) == 0) return(NULL)
    in_window <- candidates %>% filter(term_start <= qyear, term_end >= qyear)
    if (nrow(in_window) > 0) return(in_window[1, ])
    # Fall back to most recent term
    candidates %>% arrange(desc(term_start)) %>% slice(1)
  }

  # Pass 1: exact key match
  for (k in keys) {
    cands <- lookup_keys_raw %>% filter(lookup_key == k)
    m <- best_match(cands)
    if (!is.null(m))
      return(list(party   = m$party_family,
                  rs_name = m$name,
                  method  = "exact"))
  }

  # Pass 2: token overlap (>=1 non-trivial token for RS single-name MPs)
  tokens <- str_split(clean, "\\s+")[[1]]
  tokens <- tokens[!str_detect(tokens, "^[A-Z]{1,2}\\.?$") & nchar(tokens) > 2]

  if (length(tokens) >= 1) {
    # Filter to candidates sharing at least one token, then score
    cands <- lookup_keys_raw %>%
      filter(str_detect(lookup_key, fixed(tokens[length(tokens)])))
    scored <- map_dfr(seq_len(nrow(cands)), function(i) {
      ct <- str_split(cands$lookup_key[i], "\\s+")[[1]]
      tibble(
        idx       = i,
        n_overlap = sum(tokens %in% ct),
        n_union   = length(union(tokens, ct))
      )
    })
    if (nrow(scored) > 0) {
      best_idx <- scored %>% filter(n_overlap >= 1) %>%
        arrange(desc(n_overlap / n_union)) %>% slice(1) %>% pull(idx)
      if (length(best_idx) > 0) {
        m <- best_match(cands[best_idx, ])
        if (!is.null(m))
          return(list(party   = m$party_family,
                      rs_name = m$name,
                      method  = "token_overlap"))
      }
    }
  }

  # Pass 3: Jaro-Winkler fuzzy match (threshold 0.88, min key length 5)
  jw_keys   <- unique(lookup_keys_raw$lookup_key[nchar(lookup_keys_raw$lookup_key) >= 5])
  jw_scores <- stringsim(clean, jw_keys, method = "jw", p = 0.1)
  best_jw   <- which.max(jw_scores)
  if (length(best_jw) > 0 && jw_scores[best_jw] >= 0.88) {
    best_key <- jw_keys[best_jw]
    cands    <- lookup_keys_raw %>% filter(lookup_key == best_key)
    m <- best_match(cands)
    if (!is.null(m))
      return(list(party   = m$party_family,
                  rs_name = m$name,
                  method  = "fuzzy_jw"))
  }

  list(party = NA_character_, rs_name = NA_character_, method = "miss")
}

results <- vector("list", nrow(to_match))
for (i in seq_len(nrow(to_match))) {
  results[[i]] <- match_one_rs(to_match$raw_name[i], to_match$question_year[i])
}

crosswalk <- to_match %>%
  select(raw_name) %>%
  mutate(
    party_family    = map_chr(results, "party"),
    matched_rs_name = map_chr(results, "rs_name"),
    method          = map_chr(results, "method")
  )

cat("\n[M2] Match results:\n")
crosswalk %>% count(method) %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  print()

raw_matched <- rs_raw %>%
  left_join(crosswalk, by = "raw_name")

cat(sprintf("\nQuestion-level match rate: %.1f%%  (%d / %d)\n",
            100 * mean(!is.na(raw_matched$party_family)),
            sum(!is.na(raw_matched$party_family)),
            nrow(raw_matched)))
#}

# =============================================================================
# SECTION 5: Save crosswalk
# =============================================================================
#{
out_path <- file.path(INPDIR, "rs_name_crosswalk.csv")
crosswalk %>%
  filter(!is.na(party_family)) %>%
  select(raw_name, party_family, matched_rs_name, method) %>%
  write_csv(out_path)

cat(sprintf("\n[M2] Saved %d matched entries to %s\n",
            sum(!is.na(crosswalk$party_family)), out_path))

cat("\nTop 20 unmatched RS names:\n")
raw_matched %>%
  filter(is.na(party_family)) %>%
  count(raw_name, sort = TRUE) %>%
  head(20) %>%
  print(n = 20)
#}

cat("\n[M2] Done.\n")
