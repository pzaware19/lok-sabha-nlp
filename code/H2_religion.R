# =============================================================================
# H2_religion.R — Religion Proxy Analysis of Lok Sabha Questions
# Author: Piyush Zaware
# Updated: 2026-06-14
#
# PURPOSE:
#   Identifies Muslim MPs via Islamic name patterns in the parliamentary
#   record and compares their questioning behaviour to non-Muslim MPs.
#   Also flags Sikh MPs (Singh/Kaur with Punjabi first names) and
#   Christian MPs (South Indian Christian surname patterns).
#
#   LIMITATIONS (stated explicitly):
#   - Name-based inference is a lower bound. Muslim MPs without Islamic
#     name markers (e.g., some from families with non-Arabic names) are
#     not captured.
#   - "Hindu" is the residual category and cannot be positively identified.
#   - This is an approximation for research purposes, not ground truth.
#
# OUTPUTS:
#   output/figures/religion_ministry.png
#   output/figures/religion_party.png
#   output/figures/religion_vocab.png
#   output/figures/religion_tone.png
#   output/tables/religion_tags.csv
# =============================================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

pkgs <- c("arrow","tidyverse","tidytext","patchwork","ggrepel","scales","stopwords")
to_inst <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_inst)) install.packages(to_inst)
suppressPackageStartupMessages(lapply(pkgs, library, character.only = TRUE))

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")
NAVY   <- "#0D1B2A"; SAFFRON <- "#FF6B35"; GREEN <- "#138808"
PURPLE <- "#6A3D9A"; TEAL <- "#2CA25F"
MUSLIM_COL  <- "#1B7837"
SIKH_COL    <- "#762A83"
CHRIST_COL  <- "#E08214"
OTHER_COL   <- "#AAAAAA"

source(file.path(CODDIR, "._stop_words.R"))

# ============================================================
# SECTION 1: Religion tagging via name patterns
# ============================================================
#{
cat("Building religion tags...\n")

# Islamic name signals — curated for Indian parliamentary names
MUSLIM_FIRST_PAT <- regex(paste0(
  "\\b(",
  "MOHD|MOHAMMAD|MUHAMMED|MUHAMMED|MOHAMMED|MOHAMAD|MUHAMAD|",
  "ABDUL|ABUL|ABDUR|ABDUS|ABDU|",
  "SYED|SAYYAD|SAYYED|SAEED|",
  "MAULANA|MUFTI|HAJI|HAFIZ|",
  "ASADUDDIN|IMTIYAZ|GHULAM|IQBAL|ZAFAR|ASLAM|IRFAN|ARIF|",
  "RASHID|NAEEM|SALIM|SALMAN|SHAHID|TAHIR|TARIQ|WARIS|",
  "YAKUB|YUSUF|ZAHEER|ZAID|KADIR|KARIM|LATIF|MUKHTAR|",
  "SHAUKAT|NAWAB|FARRUKH|AZAM|AZIZ|AFZAL|IQBAL|ISMAIL|",
  "BADARUDDIN|BADRUDDIN|NIZAMUDDIN|SALAUDDIN|ZIAUDDIN|",
  "FASIAL|FAISAL|FURQAN|HABIBUR|HAMID|HANIF|HASANAIN|",
  "JAVED|KHURSHID|LIAQUAT|MASOOD|MEHMOOD|MINHAJ|MIRWAIZ|",
  "NOOR|OMAR|OWAISI|PARVEZ|RAFIUDDIN|RIYAZ|RUKSAR|",
  "SABIR|SALAHUDDIN|SAMAD|SHABIR|SHAFIQ|SHAKEEL|",
  "SIDDIQ|SIRAJUDDIN|TANVIR|TAUSIF|WAJID|WASEEM|ZEESHAN",
  ")\\b"
), ignore_case = TRUE)

MUSLIM_SURNAME_PAT <- regex(paste0(
  "\\b(",
  "ANSARI|SIDDIQUI|SIDDIQUE|QURESHI|QUERISHI|",
  "FAROOQUI|FAROOQI|HASHMI|BUKHARI|NAQVI|RIZVI|",
  "MANIYAR|BAIG|BEIG|MIAN|MIRZA|PATEL(?=.*MOHD)|",
  "HUSSAINI|ALVI|DEHLVI",
  ")\\b"
), ignore_case = TRUE)

# KHAN and HUSSAIN are not included in surname-only patterns
# because they appear across communities; they need a first-name co-signal
MUSLIM_CONAME_PAT <- regex("\\b(KHAN|HUSSAIN|HUSAIN|ALI)\\b", ignore_case = TRUE)

# Sikh name signals: -Singh or -Kaur is common but not sufficient alone
# (many Hindus have Singh). Combine with known Punjabi first names.
SIKH_FIRST_PAT <- regex(paste0(
  "\\b(",
  "GURJEET|GURPREET|GURINDER|GURDIP|GURDEV|GURMEET|GURNAM|",
  "HARSIMRAT|HARSIMRAN|HARDEEP|HARINDER|HARMINDER|HARPAL|",
  "AMARINDER|AMRINDER|AMRITPAL|AMRIT|",
  "SUKHBIR|SUKHDEV|SUKHJINDER|SUKHMINDER|",
  "MANPREET|MANINDER|MANMEET|MANJINDER|",
  "NAVDEEP|NAVJOT|NAVNEET|NARINDER|",
  "PARAMJIT|PARAMVIR|PARMINDER|PERMINDER|",
  "RAJINDER|RAVNEET|RAVINDER|",
  "SIMRANJIT|SATINDER|SARABJIT|",
  "JASWANT|JASWINDER|JASPAL|JASBIR|",
  "BALWANT|BALVINDER|BALDEV|BALBIR|",
  "CHARANJIT|CHARANJEET|DALJIT|DALVINDER|",
  "FATEH|INDER|INDERJIT|KULDEEP|KULWANT|LAKHWINDER",
  ")\\b"
), ignore_case = TRUE)

# South Indian Christian name signals
CHRISTIAN_PAT <- regex(paste0(
  "\\b(",
  "VARGHESE|MATHAI|MATHEW|MATHEW|KURIEN|KURIAKOSE|",
  "JOSE|JOSHY|SEBASTIAN|XAVIER|IGNATIUS|STANISLAUS|",
  "GEORGE|THOMACHAN|THARAKAN|POULOSE|PAILY|",
  "DEVASSY|DOMINIC|FRANCIS|ANTONY|ANTHONY|",
  "BENOY|BENNY|BIJU|BINOJ|BINOY|BOBBY|",
  "CHRISTOPHER|CLARENCE|CLEMENT|CLIFTON|CLETUS",
  ")\\b"
), ignore_case = TRUE)

# Apply to lookup
lookup <- read_csv(file.path(INPDIR, "mp_party_lookup.csv"),
                   show_col_types = FALSE) %>%
  mutate(
    mp_upper = str_to_upper(str_squish(mp_name)),
    mp_key   = str_to_upper(str_squish(mp_name)),

    # Muslim: strong first-name signal OR (surname + co-name like Khan/Ali)
    muslim_firstname = str_detect(mp_upper, MUSLIM_FIRST_PAT),
    muslim_surname   = str_detect(mp_upper, MUSLIM_SURNAME_PAT),
    muslim_coname    = str_detect(mp_upper, MUSLIM_CONAME_PAT),
    is_muslim        = muslim_firstname | muslim_surname |
                       (muslim_coname & muslim_firstname),

    is_sikh      = str_detect(mp_upper, SIKH_FIRST_PAT) &
                   str_detect(mp_upper, regex("\\bSINGH\\b|\\bKAUR\\b", ignore_case = TRUE)),
    is_christian = str_detect(mp_upper, CHRISTIAN_PAT),

    religion = case_when(
      is_muslim    ~ "Muslim",
      is_sikh      ~ "Sikh",
      is_christian ~ "Christian",
      TRUE         ~ "Other/Hindu"
    ),

    mp_norm = vapply(vapply(mp_key,
      function(s) {
        s <- str_remove_all(s, "\\b(SHRIMATI|SMT\\.?|KUMARI|MRS\\.?|MS\\.?|DR\\.?|PROF\\.?|SH\\.?|SHRI\\.?)\\b")
        str_squish(s)
      }, character(1)),
      function(s) {
        parts <- str_split(str_squish(s), "\\s+")[[1]]
        if (length(parts) <= 2) return(s)
        paste(parts[1], parts[length(parts)])
      }, character(1))
  ) %>%
  arrange(desc(lok_no)) %>%
  distinct(mp_norm, .keep_all = TRUE)

religion_counts <- lookup %>% count(religion, sort = TRUE)
cat("\nReligion tag distribution (MP-level):\n")
print(religion_counts)

write_csv(lookup %>% select(mp_name, mp_norm, party_family, constituency, religion,
                             is_muslim, is_sikh, is_christian),
          file.path(TABDIR, "religion_tags.csv"))
#}

# ============================================================
# SECTION 2: Load questions + attach religion tags
# ============================================================
#{
cat("\nLoading starred questions...\n")
parquet_files <- list.files(TMPDIR, pattern = "train-.*\\.parquet$", full.names = TRUE)
raw <- purrr::map_dfr(parquet_files, function(f)
  read_parquet(f, col_select = c("id","lok_no","type","ministry","members","question_text")))
starred <- raw %>% filter(type == "STARRED", lok_no >= 16)

crosswalk <- read_csv(file.path(INPDIR, "mp_name_crosswalk.csv"),
                      show_col_types = FALSE)

# Religion is derived from the matched canonical name (via regex on lookup)
religion_map <- setNames(lookup$religion, lookup$mp_name)

starred <- starred %>%
  mutate(
    primary_raw = map_chr(members, function(x)
      tryCatch(str_squish(as.character(list(x)[[1]])[1]), error = function(e) NA_character_))
  ) %>%
  left_join(crosswalk %>% select(raw_name, lok_no, party_family, matched_mp_name),
            by = c("primary_raw" = "raw_name", "lok_no")) %>%
  mutate(religion = coalesce(religion_map[matched_mp_name], "Other/Hindu")) %>%
  filter(!is.na(question_text))

# Ministry cleaning (reuse the same recode as F1)
ministry_recode <- c(
  "AGRICULTURE AND FARMERS WELFARE"="Agriculture","AGRICULTURE"="Agriculture",
  "RAILWAYS"="Railways","FINANCE"="Finance","HOME AFFAIRS"="Home Affairs",
  "HEALTH AND FAMILY WELFARE"="Health","HEALTH"="Health",
  "COMMUNICATIONS"="Communications","EDUCATION"="Education",
  "HUMAN RESOURCE DEVELOPMENT"="Education","ROAD TRANSPORT AND HIGHWAYS"="Road Transport",
  "COMMERCE AND INDUSTRY"="Commerce","SOCIAL JUSTICE AND EMPOWERMENT"="Social Justice",
  "POWER"="Power","EXTERNAL AFFAIRS"="External Affairs",
  "HOUSING AND URBAN AFFAIRS"="Urban Development",
  "WOMEN AND CHILD DEVELOPMENT"="Women & Child","LABOUR AND EMPLOYMENT"="Labour",
  "RURAL DEVELOPMENT"="Rural Development","DEFENCE"="Defence",
  "PETROLEUM AND NATURAL GAS"="Petroleum","JAL SHAKTI"="Jal Shakti",
  "WATER RESOURCES"="Jal Shakti","TRIBAL AFFAIRS"="Tribal Affairs",
  "MINORITY AFFAIRS"="Minority Affairs","TEXTILES"="Textiles",
  "SCIENCE AND TECHNOLOGY"="Science & Tech","INFORMATION AND BROADCASTING"="I&B",
  "COAL"="Coal","PANCHAYATI RAJ"="Panchayati Raj",
  "NEW AND RENEWABLE ENERGY"="Renewables","SKILL DEVELOPMENT AND ENTREPRENEURSHIP"="Skill Dev.",
  "FISHERIES, ANIMAL HUSBANDRY AND DAIRYING"="Fisheries & Animal",
  "ELECTRONICS AND INFORMATION TECHNOLOGY"="Electronics & IT",
  "ENVIRONMENT, FOREST AND CLIMATE CHANGE"="Environment",
  "LAW AND JUSTICE"="Law & Justice","CIVIL AVIATION"="Civil Aviation",
  "YOUTH AFFAIRS AND SPORTS"="Youth & Sports","TOURISM"="Tourism",
  "CHEMICALS AND FERTILIZERS"="Chemicals","CULTURE"="Culture",
  "AYUSH"="Ayush","AYURVEDA,YOGA & NATUROPATHY,UNANI,SIDDHA AND HOMEOPATHY (AYUSH)"="Ayush",
  "FOOD PROCESSING INDUSTRIES"="Food Processing","STEEL"="Steel",
  "MINES"="Mines","MICRO,SMALL AND MEDIUM ENTERPRISES"="MSME",
  "MICRO, SMALL AND MEDIUM ENTERPRISES"="MSME",
  "CONSUMER AFFAIRS, FOOD AND PUBLIC DISTRIBUTION"="Food & Consumer",
  "PORTS, SHIPPING AND WATERWAYS"="Ports & Shipping","SHIPPING"="Ports & Shipping",
  "HEAVY INDUSTRIES"="Heavy Industries"
)

starred <- starred %>%
  mutate(ministry_clean = recode(str_to_upper(str_trim(ministry)),
                                  !!!ministry_recode,
                                  .default = str_to_title(str_trim(ministry))))

q_counts <- starred %>% count(religion, name = "total_q")
cat("\nQuestion counts by religion:\n")
print(q_counts)
#}

# ============================================================
# SECTION 3: Figure 1 — Ministry targeting by religion
# ============================================================
#{
cat("\nPlotting ministry targeting...\n")

nat_share <- starred %>%
  count(ministry_clean, name = "nat_n") %>%
  mutate(nat_share = nat_n / sum(nat_n))

# Focus on Muslim vs other (largest identifiable minority)
religion_min <- starred %>%
  filter(religion %in% c("Muslim","Other/Hindu")) %>%
  group_by(religion, ministry_clean) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(religion) %>%
  mutate(share = n / sum(n)) %>%
  ungroup()

gap_min <- religion_min %>%
  pivot_wider(names_from = religion, values_from = c(n, share), values_fill = 0) %>%
  mutate(gap = `share_Muslim` - `share_Other/Hindu`) %>%
  filter(`n_Muslim` >= 2) %>%
  slice_max(abs(gap), n = 24) %>%
  mutate(
    ministry_clean = fct_reorder(ministry_clean, gap),
    direction      = if_else(gap > 0, "Muslim MPs focus more", "Muslim MPs focus less")
  )

p_religion_min <- ggplot(gap_min,
                          aes(x = gap * 100, y = ministry_clean, fill = direction)) +
  geom_col(width = 0.75) +
  geom_vline(xintercept = 0, linewidth = 0.5, colour = "grey40") +
  scale_fill_manual(
    values = c("Muslim MPs focus more" = MUSLIM_COL,
               "Muslim MPs focus less" = OTHER_COL),
    name = NULL
  ) +
  labs(
    title    = "Which ministries do Muslim MPs question more?",
    subtitle = "Percentage point difference in ministry share (Muslim MPs minus other MPs).\nPositive = Muslim MPs disproportionately target this ministry.",
    x = "Percentage point difference (Muslim minus others)", y = NULL,
    caption  = "Muslim MPs identified by Islamic name patterns. Lower bound estimate."
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", colour = NAVY, size = 12),
        plot.subtitle = element_text(colour = "grey40", size = 9),
        legend.position = "bottom")

ggsave(file.path(FIGDIR, "religion_ministry.png"),
       p_religion_min, width = 11, height = 8, dpi = 180)
cat("Saved: religion_ministry.png\n")
#}

# ============================================================
# SECTION 4: Figure 2 — Muslim MP questions by party
# ============================================================
#{
cat("Plotting religion by party...\n")

religion_party <- starred %>%
  filter(!is.na(party_family)) %>%
  group_by(party_family) %>%
  summarise(
    total    = n(),
    muslim_n = sum(religion == "Muslim"),
    sikh_n   = sum(religion == "Sikh"),
    chr_n    = sum(religion == "Christian"),
    muslim_pct = 100 * muslim_n / total,
    .groups  = "drop"
  ) %>%
  filter(total >= 30, muslim_n >= 1) %>%
  arrange(desc(muslim_pct))

p_religion_party <- religion_party %>%
  mutate(party_family = fct_reorder(party_family, muslim_pct)) %>%
  ggplot(aes(x = muslim_pct, y = party_family)) +
  geom_col(fill = MUSLIM_COL, width = 0.75, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.1f%%  (%d Qs)", muslim_pct, muslim_n)),
            hjust = -0.05, size = 3.2, colour = "grey30") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.35))) +
  labs(
    title    = "Share of starred questions from Muslim MPs, by party",
    subtitle = "% of each party's starred questions where the primary questioner\nis identified as Muslim by name pattern.",
    x = "% of party's starred questions from Muslim MPs", y = NULL,
    caption  = "Muslim identification via Islamic first names and surnames. Lower bound."
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", colour = NAVY, size = 12),
        plot.subtitle = element_text(colour = "grey40", size = 9))

ggsave(file.path(FIGDIR, "religion_party.png"),
       p_religion_party, width = 10, height = 6, dpi = 180)
cat("Saved: religion_party.png\n")
#}

# ============================================================
# SECTION 5: Figure 3 — Distinctive vocabulary of Muslim MPs
# ============================================================
#{
cat("Plotting Muslim MP vocabulary...\n")

clean_text <- function(t) {
  if (is.na(t)) return("")
  t <- str_replace_all(t, "##.*?\\n", " ")
  t <- str_replace_all(t, "\\([a-z]\\)", " ")
  str_squish(t)
}

religion_words <- starred %>%
  mutate(text_c = map_chr(question_text, clean_text),
         group  = religion) %>%
  filter(group %in% c("Muslim","Other/Hindu"), nchar(text_c) > 30) %>%
  unnest_tokens(word, text_c) %>%
  filter(!word %in% COMBINED_STOP,
         str_detect(word, "^[a-z]+$"), nchar(word) >= 5) %>%
  count(group, word)

n_muslim <- sum(religion_words$n[religion_words$group == "Muslim"])
n_other  <- sum(religion_words$n[religion_words$group == "Other/Hindu"])

log_ratio_rel <- religion_words %>%
  pivot_wider(names_from = group, values_from = n, values_fill = 0) %>%
  rename(muslim_n = Muslim, other_n = `Other/Hindu`) %>%
  filter(muslim_n + other_n >= 10, muslim_n >= 3) %>%
  mutate(
    m_rate    = (muslim_n + 0.5) / (n_muslim + 0.5),
    o_rate    = (other_n  + 0.5) / (n_other  + 0.5),
    log_ratio = log2(m_rate / o_rate)
  )

top_muslim_words <- log_ratio_rel %>%
  slice_max(log_ratio, n = 20) %>%
  mutate(word = fct_reorder(word, log_ratio))

p_religion_vocab <- top_muslim_words %>%
  ggplot(aes(x = word, y = log_ratio, fill = log_ratio)) +
  geom_col(width = 0.75, show.legend = FALSE) +
  coord_flip() +
  scale_fill_gradient(low = "#C7E9C0", high = MUSLIM_COL) +
  labs(
    title    = "Policy words Muslim MPs use more than others",
    subtitle = "Log2 ratio of word rates (Muslim vs non-Muslim MPs).\nFiltered to words appearing 10+ times total and 3+ times in Muslim MP questions.",
    x = NULL, y = "Log2 ratio (Muslim / others)"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", colour = NAVY, size = 12),
        plot.subtitle = element_text(colour = "grey40", size = 9),
        axis.text.y = element_text(size = 10))

ggsave(file.path(FIGDIR, "religion_vocab.png"),
       p_religion_vocab, width = 9, height = 7, dpi = 180)
cat("Saved: religion_vocab.png\n")
#}

# ============================================================
# SECTION 6: Figure 4 — All three religions: minority affairs focus
# ============================================================
#{
cat("Plotting Minority Affairs focus by religion...\n")

min_affairs_share <- starred %>%
  group_by(religion) %>%
  summarise(
    total    = n(),
    minority = sum(ministry_clean == "Minority Affairs", na.rm = TRUE),
    pct      = 100 * minority / total,
    .groups  = "drop"
  ) %>%
  filter(total >= 20) %>%
  mutate(
    religion    = fct_reorder(religion, pct),
    fill_colour = case_when(
      religion == "Muslim"    ~ MUSLIM_COL,
      religion == "Sikh"      ~ SIKH_COL,
      religion == "Christian" ~ CHRIST_COL,
      TRUE                    ~ OTHER_COL
    )
  )

p_minority_affairs <- ggplot(min_affairs_share,
                              aes(x = pct, y = religion, fill = religion)) +
  geom_col(width = 0.75) +
  geom_text(aes(label = sprintf("%.2f%%  (%d of %d Qs)", pct, minority, total)),
            hjust = -0.05, size = 3.3, colour = "grey30") +
  scale_fill_manual(
    values = setNames(min_affairs_share$fill_colour, min_affairs_share$religion),
    guide  = "none"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.4))) +
  labs(
    title    = "Minority Affairs Ministry: questioning rate by religion",
    subtitle = "% of each religious group's starred questions directed at the Minority Affairs Ministry.",
    x = "% of group's questions targeting Minority Affairs", y = NULL,
    caption  = "Religion identified by name pattern. Sikh and Christian counts may be lower bounds."
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", colour = NAVY, size = 12),
        plot.subtitle = element_text(colour = "grey40", size = 9))

ggsave(file.path(FIGDIR, "religion_minority_affairs.png"),
       p_minority_affairs, width = 10, height = 5, dpi = 180)
cat("Saved: religion_minority_affairs.png\n")
#}

# ============================================================
# SECTION 7: Save summary for QMD callout
# ============================================================
#{
muslim_q     <- sum(starred$religion == "Muslim")
muslim_pct   <- round(100 * muslim_q / nrow(starred), 1)
sikh_q       <- sum(starred$religion == "Sikh")
christian_q  <- sum(starred$religion == "Christian")

top_muslim_min <- gap_min %>% filter(gap > 0) %>% slice_max(gap, n = 1) %>% pull(ministry_clean)
top_avoid_min  <- gap_min %>% filter(gap < 0) %>% slice_min(gap, n = 1) %>% pull(ministry_clean)

religion_summary <- list(
  muslim_q       = muslim_q,
  muslim_pct     = muslim_pct,
  sikh_q         = sikh_q,
  christian_q    = christian_q,
  top_muslim_min = top_muslim_min,
  top_avoid_min  = top_avoid_min
)
saveRDS(religion_summary, file.path(TABDIR, "religion_summary.rds"))

cat(sprintf("\nMuslim MP questions:    %d (%.1f%%)\n", muslim_q, muslim_pct))
cat(sprintf("Sikh MP questions:      %d\n", sikh_q))
cat(sprintf("Christian MP questions: %d\n", christian_q))
cat(sprintf("Top Muslim ministry:    %s\n", top_muslim_min))
cat(sprintf("Most avoided ministry:  %s\n", top_avoid_min))
cat("\n=== H2 complete ===\n")
#}
