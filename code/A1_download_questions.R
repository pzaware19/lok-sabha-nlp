# =============================================================================
# A1_download_questions.R — Download Lok Sabha Q&A from HuggingFace
# Author: Piyush Zaware
# Last updated: 2026-06-12
#
# PURPOSE:
#   Download 150K+ parliamentary questions from the opensansad/lok-sabha-qa
#   HuggingFace dataset (parquet format, no authentication required).
#   Covers 16th LS (2014–2019), 17th LS (2019–2024), 18th LS (2024–2026).
#
# OUTPUTS:
#   $INPDIR/questions_raw.rds   — full dataset (150K rows)
#   $INPDIR/questions_raw.csv   — same, CSV format
# =============================================================================

library(arrow)
library(tidyverse)

# ============================================================
# SECTION 1: Download 5 parquet files
# ============================================================
#{

base_url <- "https://huggingface.co/datasets/opensansad/lok-sabha-qa/resolve/main/data"

files <- paste0("train-0000", 0:4, "-of-00005.parquet")

cat("Downloading", length(files), "parquet files from HuggingFace...\n")

questions_list <- purrr::map(seq_along(files), function(i) {
  url      <- paste0(base_url, "/", files[i])
  dest     <- file.path(TMPDIR, files[i])

  if (!file.exists(dest)) {
    cat(" Downloading", files[i], "...")
    tryCatch(
      download.file(url, dest, mode = "wb", quiet = TRUE),
      error = function(e) cat(" FAILED:", conditionMessage(e), "\n")
    )
    cat(" done\n")
  } else {
    cat(" Using cached:", files[i], "\n")
  }

  if (file.exists(dest)) read_parquet(dest) else NULL
})

questions_raw <- bind_rows(questions_list)
cat("\nTotal records downloaded:", nrow(questions_raw), "\n")
cat("Columns:", paste(names(questions_raw), collapse = ", "), "\n")

#}

# ============================================================
# SECTION 2: Inspect and clean
# ============================================================
#{

cat("\nLok Sabha coverage:\n")
questions_raw %>%
  dplyr::count(lok_no, type) %>%
  pivot_wider(names_from = type, values_from = n, values_fill = 0) %>%
  print()

cat("\nDate range:\n")
cat(" Earliest:", min(questions_raw$date, na.rm = TRUE), "\n")
cat(" Latest:  ", max(questions_raw$date, na.rm = TRUE), "\n")

cat("\nMissing text:\n")
cat(" question_text NA:", sum(is.na(questions_raw$question_text)), "\n")
cat(" members NA:      ", sum(is.na(questions_raw$members)),       "\n")

# Parse members field (stored as list-column in parquet)
# Each row can have multiple MPs as co-signatories
questions_clean <- questions_raw %>%
  mutate(
    # Convert list-column to character (first member = primary MP)
    members_str = purrr::map_chr(members, function(m) {
      if (is.null(m) || length(m) == 0) return(NA_character_)
      paste(m, collapse = "; ")
    }),
    primary_mp = purrr::map_chr(members, function(m) {
      if (is.null(m) || length(m) == 0) return(NA_character_)
      m[[1]]
    }),
    # Parse date
    date_parsed = as.Date(date)
  ) %>%
  filter(!is.na(question_text), nchar(question_text) > 50)

cat("\nAfter cleaning:", nrow(questions_clean), "questions\n")

#}

# ============================================================
# SECTION 3: Save
# ============================================================
#{

saveRDS(questions_clean, file.path(INPDIR, "questions_raw.rds"))
write_csv(questions_clean %>% select(-members),
          file.path(INPDIR, "questions_raw.csv"))

cat("\nA1 complete. Saved to", INPDIR, "\n")

#}
