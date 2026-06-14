# ._stop_words.R — Shared stop words for all word-frequency analyses
# Author: Piyush Zaware
#
# PURPOSE:
#   Builds three stop-word lists and exports them to the calling environment:
#
#   PARLIAMENT_STOP  — parliamentary boilerplate + English function words
#   MP_NAME_STOP     — individual name tokens from mp_party_lookup.csv that
#                      fail an English dictionary check (so "green" is NOT
#                      blocked even though some MP is named Green, but "singh",
#                      "mahadik", "dhananjay", "supriya" etc. are blocked)
#   COMBINED_STOP    — union of the above; use this in filter() calls
#
# REQUIRES: INPDIR must already be defined in the calling environment.
# SOURCE this file, do not run standalone.

stopifnot(exists("INPDIR"))

if (!requireNamespace("hunspell",   quietly = TRUE)) install.packages("hunspell")
if (!requireNamespace("stopwords",  quietly = TRUE)) install.packages("stopwords")
suppressPackageStartupMessages({
  library(hunspell)
  library(stopwords)
})

# ── 1. Parliamentary boilerplate ─────────────────────────────────────────────
PARLIAMENT_STOP <- unique(c(
  # Parliamentary formulaic phrases
  "will","minister","whether","government","please","state","details",
  "thereof","taken","steps","also","further","said","country","india",
  "hon","aware","regard","provide","information","thereon","thereunder",
  "proposed","considered","members","question","starred","unstarred",
  "asked","since","being","given","refer","statement","fact","reply",
  "part","crore","lakh","rupees","year","years","number","total","list",
  "during","wherein","thereof","hereto","under","above","below","same",
  "parliament","session","rajya","sabha","house","minister","ministry",
  "shri","smt","kumari","shrimati",
  # Standard English stop words
  stopwords::stopwords("en")
))

# ── 2. MP name blocklist ──────────────────────────────────────────────────────
# Strategy: extract every individual token from MP full names + constituency
# names in the lookup CSV. Keep only tokens that fail the English dictionary
# (so valid English words that happen to also be surnames are NOT blocked).

lookup_raw <- tryCatch(
  readr::read_csv(file.path(INPDIR, "mp_party_lookup.csv"),
                  show_col_types = FALSE),
  error = function(e) NULL
)

MP_NAME_STOP <- character(0)

if (!is.null(lookup_raw)) {
  # Name tokens from mp_name column
  name_tokens <- lookup_raw$mp_name %>%
    stringr::str_to_lower() %>%
    # Strip honorifics before splitting
    stringr::str_remove_all(
      "\\b(shrimati|smt\\.?|kumari|mrs\\.?|ms\\.?|dr\\.?|prof\\.?|sh\\.?|shri\\.?)\\b"
    ) %>%
    stringr::str_squish() %>%
    stringr::str_split("\\s+") %>%
    unlist()

  # Constituency name tokens (strip "(SC)", "(ST)", digits, punctuation)
  const_tokens <- lookup_raw$constituency %>%
    stringr::str_to_lower() %>%
    stringr::str_remove_all("\\(sc\\)|\\(st\\)|\\d+|[^a-z\\s]") %>%
    stringr::str_squish() %>%
    stringr::str_split("\\s+") %>%
    unlist()

  all_tokens <- unique(c(name_tokens, const_tokens))
  all_tokens <- all_tokens[nchar(all_tokens) >= 4]   # skip very short tokens
  all_tokens <- all_tokens[!is.na(all_tokens) & all_tokens != ""]

  # Only block tokens that are NOT valid English words.
  # Checked against both en_US and en_GB to handle British spellings.
  en_us <- tryCatch(hunspell::hunspell_check(all_tokens,
                      dict = hunspell::dictionary("en_US")),
                    error = function(e) rep(FALSE, length(all_tokens)))
  en_gb <- tryCatch(hunspell::hunspell_check(all_tokens,
                      dict = hunspell::dictionary("en_GB")),
                    error = function(e) rep(FALSE, length(all_tokens)))
  is_english <- en_us | en_gb

  MP_NAME_STOP <- all_tokens[!is_english]

  message(sprintf(
    "[._stop_words] MP name blocklist: %d tokens (from %d MPs, %d constituencies)",
    length(MP_NAME_STOP),
    nrow(lookup_raw),
    n_distinct(lookup_raw$constituency)
  ))
}

# ── 3. Combined stop-word list ────────────────────────────────────────────────
COMBINED_STOP <- unique(c(PARLIAMENT_STOP, MP_NAME_STOP))

message(sprintf("[._stop_words] COMBINED_STOP: %d words", length(COMBINED_STOP)))
