# =============================================================================
# A2_get_party_data.R — Get MP → Party Mapping (Wikipedia)
# Author: Piyush Zaware
# Last updated: 2026-06-12
#
# PURPOSE:
#   Scrape Wikipedia's "List of members of the Nth Lok Sabha" pages.
#   Each page has state-wise tables: Constituency | Name | Party.
#   Build MP → party lookup for 16th, 17th, 18th Lok Sabha.
#   Merge into the questions dataset.
#
# OUTPUTS:
#   $INPDIR/mp_party_lookup.csv        — MP name → party (all 3 LSs)
#   $INPDIR/questions_with_party.rds   — questions with party_family column
# =============================================================================

library(rvest)
library(tidyverse)

# ============================================================
# SECTION 1: Scrape Wikipedia member lists
# ============================================================
#{

scrape_ls_members <- function(lok_no) {
  url <- paste0(
    "https://en.wikipedia.org/wiki/List_of_members_of_the_",
    lok_no, switch(as.character(lok_no),
      "16" = "th", "17" = "th", "18" = "th"), "_Lok_Sabha"
  )
  cat("Scraping LS", lok_no, "from Wikipedia...\n")

  page <- tryCatch(read_html(url), error = function(e) {
    cat("  Failed:", conditionMessage(e), "\n"); NULL
  })
  if (is.null(page)) return(NULL)

  # All tables on the page (one per state)
  tables <- page %>% html_nodes("table.wikitable") %>% html_table(fill = TRUE)
  cat("  Tables found:", length(tables), "\n")

  # Standard table: No. | Constituency | Member | Party(icon) | Party(name)
  # Column 4 is the flag/icon cell (usually empty string in parsed text).
  # Column 5 (or the last "party"-named column) has the actual party text.
  # Some tables have 4 columns — those have no party-name column; skip them.
  parsed <- purrr::map_dfr(tables, function(tbl) {
    if (ncol(tbl) < 5) return(NULL)

    col_lower <- tolower(trimws(names(tbl)))

    # Prefer the last column whose header contains "party" or "alliance"
    party_cols <- which(grepl("party|alliance", col_lower))
    party_col  <- if (length(party_cols) >= 1) tail(party_cols, 1) else 5

    # Name column: look for "member" or "name" in header, else column 3
    name_cols <- which(grepl("^member$|^name$|^members$", col_lower))
    name_col  <- if (length(name_cols) >= 1) name_cols[1] else 3

    tibble(
      mp_name      = as.character(tbl[[name_col]]),
      party        = as.character(tbl[[party_col]]),
      constituency = as.character(tbl[[2]])
    ) %>%
      filter(
        !is.na(mp_name), !is.na(party),
        nchar(mp_name) > 2,
        nchar(party) > 2,
        !grepl("^Name$|^Member|Constituency|Vacant|^No\\.$|^[0-9]+$",
               mp_name, ignore.case = TRUE),
        !grepl("^Party$|^Alliance$|^NA$",
               party, ignore.case = TRUE)
      )
  })

  if (nrow(parsed) == 0) {
    cat("  Warning: No valid rows for LS", lok_no, "\n")
    return(NULL)
  }

  parsed %>%
    mutate(
      lok_no  = lok_no,
      mp_name = str_squish(toupper(mp_name)),
      party   = str_squish(party)
    ) %>%
    distinct(mp_name, lok_no, .keep_all = TRUE)
}

ls_members <- purrr::map_dfr(c(16, 17, 18), scrape_ls_members)

cat("\nTotal MP-party rows scraped:", nrow(ls_members), "\n")
cat("Unique MPs:", n_distinct(ls_members$mp_name), "\n")
cat("\nTop parties:\n")
ls_members %>% dplyr::count(party, sort=TRUE) %>% head(15) %>% print()

#}

# ============================================================
# SECTION 2: Classify party families
# ============================================================
#{

ls_members <- ls_members %>%
  mutate(
    party_family = case_when(
      grepl("Bharatiya Janata|BJP",                  party, ignore.case=TRUE) ~ "BJP",
      grepl("Indian National Congress|INC|Congress", party, ignore.case=TRUE) ~ "INC",
      grepl("Communist|CPI|CPM|CPI.M",               party, ignore.case=TRUE) ~ "Left",
      grepl("Bahujan Samaj|BSP",                     party, ignore.case=TRUE) ~ "BSP",
      grepl("Aam Aadmi|AAP",                         party, ignore.case=TRUE) ~ "AAP",
      grepl("Trinamool|TMC|AITC",                    party, ignore.case=TRUE) ~ "TMC",
      grepl("Samajwadi|SP$",                         party, ignore.case=TRUE) ~ "SP",
      grepl("Nationalist Congress|NCP",              party, ignore.case=TRUE) ~ "NCP",
      grepl("Shiv Sena",                             party, ignore.case=TRUE) ~ "Shiv Sena",
      grepl("Telugu Desam|TDP",                      party, ignore.case=TRUE) ~ "TDP",
      grepl("Janata Dal.*United|JD.U|JDU",           party, ignore.case=TRUE) ~ "JDU",
      grepl("Rashtriya Janata|RJD",                  party, ignore.case=TRUE) ~ "RJD",
      grepl("Dravida Munnetra|DMK",                  party, ignore.case=TRUE) ~ "DMK",
      grepl("AIADMK|Anna Dravida",                   party, ignore.case=TRUE) ~ "AIADMK",
      grepl("Independent",                           party, ignore.case=TRUE) ~ "Independent",
      TRUE ~ "Regional"
    )
  )

write_csv(ls_members, file.path(INPDIR, "mp_party_lookup.csv"))
cat("\nParty families:\n")
ls_members %>% dplyr::count(party_family, sort=TRUE) %>% print()

#}

# ============================================================
# SECTION 3: Merge into questions
# ============================================================
#{

questions <- readRDS(file.path(INPDIR, "questions_raw.rds"))

questions <- questions %>%
  mutate(primary_mp_upper = str_squish(toupper(primary_mp)))

questions_party <- questions %>%
  left_join(
    ls_members %>% select(mp_name, party, party_family, lok_no),
    by = c("primary_mp_upper" = "mp_name", "lok_no")
  )

match_rate <- mean(!is.na(questions_party$party))
cat("Party match rate:", round(match_rate * 100, 1), "%\n")

# Flag unmatched as Unknown
questions_party <- questions_party %>%
  mutate(party_family = replace_na(party_family, "Unknown"))

cat("\nQuestions by party family:\n")
questions_party %>%
  filter(party_family != "Unknown") %>%
  dplyr::count(party_family, sort = TRUE) %>%
  print()

saveRDS(questions_party, file.path(INPDIR, "questions_with_party.rds"))
cat("\nA2 complete.\n")

#}
