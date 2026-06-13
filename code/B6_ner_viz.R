# =============================================================================
# B6_ner_viz.R — Named Entity Recognition Visualization
# Author: Piyush Zaware
# Updated: 2026-06-13
#
# PURPOSE:
#   Visualizes spaCy NER outputs from B4. Key findings:
#   - Geographic (GPE): regional parties concentrate on home states;
#     national parties spread more uniformly
#   - Person: which parties name politicians / public figures most
#   - ORG: institutional focus by party
#
# INPUTS:
#   output/tables/ner_doc.csv
#   output/tables/ner_gpe_party.csv
#   output/tables/ner_person_party.csv
#   output/tables/ner_org_party.csv
#
# OUTPUTS:
#   output/figures/ner_state_heatmap.png
#   output/figures/ner_state_concentration.png
#   output/figures/ner_top_persons.png
#   output/figures/ner_top_orgs.png
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(ggrepel)
})

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")

PARTY_COLORS <- c(
  BJP        = "#FF6B35",
  INC        = "#1F78B4",
  Left       = "#E31A1C",
  BSP        = "#6A3D9A",
  AAP        = "#33A02C",
  TMC        = "#B15928",
  SP         = "#FF7F00",
  JDU        = "#A6CEE3",
  DMK        = "#B2DF8A",
  TDP        = "#FDBF6F",
  `Shiv Sena`= "#FB9A99",
  RJD        = "#CAB2D6",
  NCP        = "#999999",
  Regional   = "#CCCCCC"
)

INDIA_STATES <- c(
  "ANDHRA PRADESH", "ARUNACHAL PRADESH", "ASSAM", "BIHAR", "CHHATTISGARH",
  "GOA", "GUJARAT", "HARYANA", "HIMACHAL PRADESH", "JHARKHAND", "KARNATAKA",
  "KERALA", "MADHYA PRADESH", "MAHARASHTRA", "MANIPUR", "MEGHALAYA",
  "MIZORAM", "NAGALAND", "ODISHA", "PUNJAB", "RAJASTHAN", "SIKKIM",
  "TAMIL NADU", "TELANGANA", "TRIPURA", "UTTAR PRADESH", "UTTARAKHAND",
  "WEST BENGAL", "DELHI", "JAMMU AND KASHMIR", "PUDUCHERRY"
)

# ============================================================
# SECTION 1: State × party heatmap (share_pct within party)
# ============================================================
# Use ner_doc states_mentioned column: it uses exact INDIA_STATES set matching,
# which is more reliable than spaCy NER tags for Indian state names.
# The ner_gpe_party.csv keeps only top-60 global GPEs, which can cut states
# that are rare globally but important for a specific party (e.g. Tamil Nadu for DMK).
#{

ner_doc <- read_csv(file.path(TABDIR, "ner_doc.csv"), show_col_types = FALSE)

# Explode states_mentioned (semicolon-separated) to one row per state mention
state_party_raw <- ner_doc %>%
  filter(!is.na(states_mentioned), states_mentioned != "") %>%
  mutate(states_mentioned = str_split(states_mentioned, ";")) %>%
  unnest(states_mentioned) %>%
  mutate(entity = str_trim(states_mentioned)) %>%
  filter(entity != "", entity != "nan") %>%
  count(party_family, entity, name = "n_questions")

party_totals_st <- ner_doc %>%
  count(party_family, name = "party_total")

state_party <- state_party_raw %>%
  left_join(party_totals_st, by = "party_family") %>%
  mutate(share_pct = 100 * n_questions / party_total) %>%
  filter(party_family %in% names(PARTY_COLORS),
         n_questions >= 2)

# Top 15 states by total mentions across parties
top_states <- state_party %>%
  group_by(entity) %>%
  summarise(total = sum(n_questions), .groups = "drop") %>%
  slice_max(total, n = 15) %>%
  pull(entity)

# Parties with enough matched questions
major_parties <- c("BJP", "INC", "Left", "Regional", "DMK", "TDP", "JDU",
                   "SP", "Shiv Sena", "RJD", "BSP")

state_heat <- state_party %>%
  filter(entity %in% top_states,
         party_family %in% major_parties) %>%
  complete(party_family, entity, fill = list(share_pct = 0, n_questions = 0))

p_state_heat <- ggplot(state_heat,
                       aes(x = party_family,
                           y = reorder(str_to_title(entity), n_questions, sum),
                           fill = share_pct)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  geom_text(aes(label = if_else(share_pct >= 0.5,
                                sprintf("%.1f", share_pct), "")),
            size = 2.5, colour = "grey20") +
  scale_fill_distiller(palette = "YlOrRd", direction = 1,
                       name = "% of party's\nquestions") +
  labs(
    title    = "Geographic focus: which state does each party mention most?",
    subtitle = "% of party's matched starred questions that name the state. Labels shown for shares ≥ 0.5%.",
    x = NULL, y = NULL,
    caption  = "Top 15 most-mentioned states. NER extracted via spaCy en_core_web_sm."
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x  = element_text(angle = 35, hjust = 1),
    plot.title   = element_text(face = "bold"),
    legend.position = "right"
  )

ggsave(file.path(FIGDIR, "ner_state_heatmap.png"),
       p_state_heat, width = 12, height = 7, dpi = 180)
cat("Saved: ner_state_heatmap.png\n")
#}

# ============================================================
# SECTION 2: Geographic concentration by party (HHI on states)
# ============================================================
#{

state_conc <- state_party %>%
  filter(party_family %in% major_parties) %>%
  group_by(party_family) %>%
  mutate(state_share = n_questions / sum(n_questions)) %>%
  summarise(
    hhi_state    = sum(state_share^2),
    top_state    = entity[which.max(n_questions)],
    top_share    = max(state_share),
    n_states     = n_distinct(entity),
    total_q      = first(party_total),
    .groups      = "drop"
  ) %>%
  filter(total_q >= 20) %>%
  arrange(desc(hhi_state))

p_geo_conc <- ggplot(state_conc,
                     aes(x = reorder(party_family, -hhi_state),
                         y = hhi_state,
                         fill = party_family)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(aes(label = paste0(str_to_title(top_state), "\n",
                               round(100 * top_share), "%")),
            vjust = -0.3, size = 2.7, lineheight = 0.9) +
  scale_fill_manual(values = PARTY_COLORS, na.value = "#AAAAAA") +
  scale_y_continuous(
    limits = c(0, max(state_conc$hhi_state) * 1.35),
    labels = scales::number_format(accuracy = 0.01)
  ) +
  labs(
    title    = "Geographic concentration of parliamentary questions by party",
    subtitle = "HHI over Indian states mentioned. Higher = questions concentrate on fewer states.",
    x = NULL, y = "HHI (state mention concentration)",
    caption  = "Label = top state and its share of party's state mentions. Parties with 20+ matched questions."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title  = element_text(face = "bold"),
    axis.text.x = element_text(angle = 30, hjust = 1)
  )

ggsave(file.path(FIGDIR, "ner_state_concentration.png"),
       p_geo_conc, width = 10, height = 5.5, dpi = 180)
cat("Saved: ner_state_concentration.png\n")
cat("\nGeographic concentration (HHI, top states):\n")
print(state_conc %>% select(party_family, hhi_state, top_state, top_share,
                             n_states, total_q), n = Inf)
#}

# ============================================================
# SECTION 3: Top persons named by each party
# ============================================================
#{

person_party <- read_csv(file.path(TABDIR, "ner_person_party.csv"),
                         show_col_types = FALSE)

# Top 5 persons per major party
top_persons <- person_party %>%
  filter(party_family %in% major_parties,
         n_questions >= 2) %>%
  group_by(party_family) %>%
  slice_max(n_questions, n = 5) %>%
  ungroup() %>%
  mutate(entity = str_to_title(entity))

if (nrow(top_persons) > 0) {
  p_persons <- ggplot(top_persons,
                      aes(x = reorder_within(entity, n_questions, party_family),
                          y = n_questions,
                          fill = party_family)) +
    geom_col(show.legend = FALSE, width = 0.7) +
    coord_flip() +
    facet_wrap(~ party_family, scales = "free_y", ncol = 4) +
    tidytext::scale_x_reordered() +
    scale_fill_manual(values = PARTY_COLORS, na.value = "#AAAAAA") +
    labs(
      title    = "People named in starred questions, by party",
      subtitle = "Top 5 individuals per party by number of questions mentioning them (spaCy PERSON entities).",
      x = NULL, y = "Number of questions",
      caption  = "Boilerplate titles (Minister, Speaker, etc.) excluded."
    ) +
    theme_minimal(base_size = 9) +
    theme(
      plot.title  = element_text(face = "bold"),
      strip.text  = element_text(face = "bold", size = 9),
      axis.text.y = element_text(size = 8)
    )

  ggsave(file.path(FIGDIR, "ner_top_persons.png"),
         p_persons, width = 14, height = 10, dpi = 180)
  cat("Saved: ner_top_persons.png\n")
}
#}

# ============================================================
# SECTION 4: Top organisations named by each party
# ============================================================
#{

org_party <- read_csv(file.path(TABDIR, "ner_org_party.csv"),
                      show_col_types = FALSE)

# Common noisy ORG entities to filter (parliamentary boilerplate)
ORG_STOPLIST <- c(
  "LOK SABHA", "RAJYA SABHA", "PARLIAMENT", "THE GOVERNMENT", "GOVERNMENT",
  "UNION", "MINISTRY", "STATE GOVERNMENT", "CENTRAL GOVERNMENT",
  "THE MINISTRY", "STATE GOVERNMENTS", "NITI AAYOG"
)

top_orgs <- org_party %>%
  filter(party_family %in% major_parties,
         !entity %in% ORG_STOPLIST,
         n_questions >= 2) %>%
  group_by(party_family) %>%
  slice_max(n_questions, n = 5) %>%
  ungroup() %>%
  mutate(entity = str_to_title(entity))

if (nrow(top_orgs) > 0 && n_distinct(top_orgs$party_family) >= 2) {
  p_orgs <- ggplot(top_orgs,
                   aes(x = reorder_within(entity, n_questions, party_family),
                       y = n_questions,
                       fill = party_family)) +
    geom_col(show.legend = FALSE, width = 0.7) +
    coord_flip() +
    facet_wrap(~ party_family, scales = "free_y", ncol = 4) +
    tidytext::scale_x_reordered() +
    scale_fill_manual(values = PARTY_COLORS, na.value = "#AAAAAA") +
    labs(
      title    = "Organisations named in starred questions, by party",
      subtitle = "Top 5 organisations per party (spaCy ORG entities; boilerplate terms excluded).",
      x = NULL, y = "Number of questions"
    ) +
    theme_minimal(base_size = 9) +
    theme(
      plot.title  = element_text(face = "bold"),
      strip.text  = element_text(face = "bold", size = 9),
      axis.text.y = element_text(size = 8)
    )

  ggsave(file.path(FIGDIR, "ner_top_orgs.png"),
         p_orgs, width = 14, height = 10, dpi = 180)
  cat("Saved: ner_top_orgs.png\n")
}
#}

cat("\n=== B6 complete ===\n")
