# =============================================================================
# C1_ministry_analysis.R — Ministry Targeting Analysis
# Author: Piyush Zaware
# Updated: 2026-06-12
#
# PURPOSE:
#   Goes beyond topic models to ask: which parties target which ministries,
#   and is that targeting strategic?
#
#   Parties in opposition have limited levers — questioning ministries is one
#   of them. But they face a choice: spread fire broadly, or concentrate on
#   high-profile ministries? Regional parties have a different calculus:
#   they target ministries that control transfers relevant to their state.
#
#   Analysis:
#   (1) Raw party × ministry question counts
#   (2) Excess targeting: how much more (or less) does each party question
#       a ministry than we would expect under proportional questioning?
#       Expected = (party share of total questions) × (ministry total questions)
#   (3) Opposition vs government targeting patterns
#   (4) Ministry concentration (Herfindahl index per party): do some parties
#       focus on a narrow set of ministries?
#   (5) Temporal heatmap: has ministry targeting shifted across Lok Sabhas?
#   (6) Network: bipartite party-ministry graph
#
# INPUTS:
#   tmp/train-*.parquet          — raw questions (from A1)
#   tmp/doc_meta_mp.csv          — MP → party mapping (from A2/A3)
#   code/A9_incumbency_robustness.R incumbency_tbl (defined inline here)
#
# OUTPUTS:
#   output/figures/ministry_excess_heatmap.png
#   output/figures/ministry_concentration.png
#   output/figures/ministry_temporal.png
#   output/figures/ministry_network.png
#   output/tables/ministry_party_counts.csv
#   output/tables/ministry_excess_targeting.csv
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(patchwork)
  library(ggrepel)
})

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")

# ============================================================
# SECTION 1: Load and merge raw questions with party labels
# ============================================================
#{

cat("Loading parquet files...\n")
parquet_files <- list.files(TMPDIR, pattern = "train-.*\\.parquet$", full.names = TRUE)

raw <- purrr::map_dfr(parquet_files, function(f) {
  read_parquet(f, col_select = c("id", "lok_no", "session_no", "type",
                                  "ministry", "members"))
})

cat("  Total rows:", nrow(raw), "\n")
starred <- raw %>% filter(type == "STARRED")
cat("  Starred questions:", nrow(starred), "\n")

# Load MP → party mapping
mp_meta <- read_csv(file.path(TMPDIR, "doc_meta_mp.csv"), show_col_types = FALSE)

# Normalize names for matching
norm_name <- function(x) str_squish(str_to_upper(str_trim(x)))

# members is a list column (multiple MPs per question) — extract first element
starred <- starred %>%
  mutate(
    primary_member = norm_name(sapply(members, function(x) x[[1]]))
  )

# Build lookup: doc_mp is already uppercase in doc_meta_mp.csv
mp_lookup <- mp_meta %>%
  mutate(key = norm_name(doc_mp)) %>%
  select(key, party_family) %>%
  distinct()

starred <- starred %>%
  left_join(mp_lookup, by = c("primary_member" = "key"))

matched <- starred %>% filter(!is.na(party_family))
cat("  Matched to party:", nrow(matched),
    sprintf("(%.1f%%)\n", 100 * nrow(matched) / nrow(starred)))
#}

# ============================================================
# SECTION 2: Clean and consolidate ministries
# ============================================================
#{

# Ministry names are inconsistent — consolidate common variants
ministry_recode <- c(
  "HOME AFFAIRS"               = "Home Affairs",
  "MINISTRY OF HOME AFFAIRS"   = "Home Affairs",
  "FINANCE"                    = "Finance",
  "MINISTRY OF FINANCE"        = "Finance",
  "RAILWAYS"                   = "Railways",
  "MINISTRY OF RAILWAYS"       = "Railways",
  "DEFENCE"                    = "Defence",
  "MINISTRY OF DEFENCE"        = "Defence",
  "AGRICULTURE AND FARMERS WELFARE" = "Agriculture",
  "AGRICULTURE"                = "Agriculture",
  "ROAD TRANSPORT AND HIGHWAYS"= "Road Transport",
  "EXTERNAL AFFAIRS"           = "External Affairs",
  "HEALTH AND FAMILY WELFARE"  = "Health",
  "HEALTH"                     = "Health",
  "EDUCATION"                  = "Education",
  "HUMAN RESOURCE DEVELOPMENT" = "Education",
  "LABOUR AND EMPLOYMENT"      = "Labour",
  "JAL SHAKTI"                 = "Jal Shakti",
  "WATER RESOURCES"            = "Jal Shakti",
  "RURAL DEVELOPMENT"          = "Rural Development",
  "TRIBAL AFFAIRS"             = "Tribal Affairs",
  "WOMEN AND CHILD DEVELOPMENT"= "Women & Child",
  "ENVIRONMENT"                = "Environment",
  "PETROLEUM AND NATURAL GAS"  = "Petroleum",
  "POWER"                      = "Power",
  "COMMUNICATIONS"             = "Communications",
  "COMMERCE AND INDUSTRY"      = "Commerce",
  "TEXTILES"                   = "Textiles",
  "SOCIAL JUSTICE AND EMPOWERMENT" = "Social Justice",
  "MINORITY AFFAIRS"           = "Minority Affairs",
  "SCIENCE AND TECHNOLOGY"     = "Science & Tech",
  "INFORMATION AND BROADCASTING" = "I&B",
  "URBAN DEVELOPMENT"          = "Urban Development",
  "HOUSING AND URBAN AFFAIRS"  = "Urban Development",
  "STEEL"                      = "Steel",
  "MINES"                      = "Mines"
)

matched <- matched %>%
  mutate(
    ministry_clean = recode(str_to_upper(str_trim(ministry)),
                            !!!ministry_recode,
                            .default = str_to_title(str_trim(ministry)))
  )

# Keep only ministries with enough questions for reliable estimates
min_count <- 30
ministry_counts <- count(matched, ministry_clean) %>% filter(n >= min_count)
matched_filtered <- matched %>%
  filter(ministry_clean %in% ministry_counts$ministry_clean)

cat("Ministries with >= 30 questions:", nrow(ministry_counts), "\n")
cat("Questions in analysis:", nrow(matched_filtered), "\n")
#}

# ============================================================
# SECTION 3: Raw counts and excess targeting
# ============================================================
#{

# Raw party × ministry matrix
pm_counts <- matched_filtered %>%
  count(party_family, ministry_clean) %>%
  complete(party_family, ministry_clean, fill = list(n = 0))

# Expected counts under null of proportional questioning:
# E[party p, ministry m] = (total questions by p) × (total questions to m) / grand total
party_totals    <- pm_counts %>% group_by(party_family)   %>% summarise(p_total = sum(n))
ministry_totals <- pm_counts %>% group_by(ministry_clean) %>% summarise(m_total = sum(n))
grand_total     <- sum(pm_counts$n)

pm_excess <- pm_counts %>%
  left_join(party_totals,    by = "party_family") %>%
  left_join(ministry_totals, by = "ministry_clean") %>%
  mutate(
    expected = p_total * m_total / grand_total,
    excess   = n - expected,
    excess_pct = 100 * (n - expected) / pmax(expected, 1),
    # Standardized: (observed - expected) / sqrt(expected) ~ Pearson residual
    pearson_resid = (n - expected) / sqrt(pmax(expected, 0.5))
  )

write_csv(pm_counts,  file.path(TABDIR, "ministry_party_counts.csv"))
write_csv(pm_excess,  file.path(TABDIR, "ministry_excess_targeting.csv"))
cat("Saved: ministry_party_counts.csv, ministry_excess_targeting.csv\n")
#}

# ============================================================
# SECTION 4: Figure — Excess targeting heatmap (Pearson residuals)
# ============================================================
#{

# Focus on parties with enough questions
major_parties <- party_totals %>% filter(p_total >= 50) %>% pull(party_family)

p_excess <- pm_excess %>%
  filter(party_family %in% major_parties) %>%
  mutate(
    resid_capped = pmax(pmin(pearson_resid, 8), -8),  # cap for colour scale
    party_family = factor(party_family,
                          levels = major_parties[order(
                            pm_counts %>%
                              filter(party_family %in% major_parties) %>%
                              group_by(party_family) %>% summarise(tot = sum(n)) %>%
                              arrange(desc(tot)) %>% pull(party_family)
                          )])
  ) %>%
  ggplot(aes(x = party_family, y = reorder(ministry_clean, m_total),
             fill = resid_capped)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  scale_fill_gradient2(
    low      = "#1F78B4",
    mid      = "#F5F5F5",
    high     = "#B5440E",
    midpoint = 0,
    limits   = c(-8, 8),
    name     = "Pearson\nresidual"
  ) +
  labs(
    title    = "Which parties question which ministries — relative to their size",
    subtitle = "Pearson residual = (observed - expected) / sqrt(expected). Red = over-questions, blue = under-questions.",
    x = NULL, y = NULL,
    caption  = "Expected counts under the null that each party questions all ministries proportionally to their total volume."
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x  = element_text(angle = 35, hjust = 1, size = 8.5),
    axis.text.y  = element_text(size = 8.5),
    plot.title   = element_text(face = "bold"),
    legend.position = "right"
  )

ggsave(file.path(FIGDIR, "ministry_excess_heatmap.png"),
       p_excess, width = 13, height = 9, dpi = 180)
cat("Saved: ministry_excess_heatmap.png\n")
#}

# ============================================================
# SECTION 5: Ministry concentration (Herfindahl index per party)
# ============================================================
# Does each party spread its questions across many ministries,
# or concentrate on a few? HHI = sum of squared shares.
# HHI close to 1 = all questions go to one ministry (very concentrated).
# HHI close to 0 = uniform spread across all ministries.
#{

hhi <- pm_counts %>%
  left_join(party_totals, by = "party_family") %>%
  mutate(share = n / p_total) %>%
  group_by(party_family) %>%
  summarise(
    hhi          = sum(share^2),
    n_ministries = sum(n > 0),
    total_q      = first(p_total),
    top_ministry = ministry_clean[which.max(n)],
    top_share    = max(share),
    .groups      = "drop"
  ) %>%
  filter(total_q >= 30) %>%
  arrange(desc(hhi))

p_hhi <- ggplot(hhi, aes(x = reorder(party_family, -hhi),
                          y = hhi, fill = hhi)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = paste0(top_ministry, "\n", round(100*top_share), "%")),
            vjust = -0.3, size = 2.6, lineheight = 0.9) +
  scale_fill_gradient(low = "#A6CEE3", high = "#B5440E",
                      name = "HHI", guide = "none") +
  scale_y_continuous(limits = c(0, max(hhi$hhi) * 1.25),
                     labels = scales::number_format(accuracy = 0.01)) +
  labs(
    title    = "Ministry concentration by party",
    subtitle = "Herfindahl-Hirschman Index of question share across ministries.\nHigher = more concentrated. Label shows top ministry and its share.",
    x = NULL, y = "HHI (ministry concentration)",
    caption  = "Only parties with 30+ starred questions included."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title  = element_text(face = "bold"),
    axis.text.x = element_text(angle = 30, hjust = 1)
  )

ggsave(file.path(FIGDIR, "ministry_concentration.png"),
       p_hhi, width = 10, height = 5.5, dpi = 180)
cat("Saved: ministry_concentration.png\n")

cat("\nMinistry concentration (HHI):\n")
print(hhi %>% select(party_family, hhi, n_ministries, top_ministry, top_share,
                     total_q), n = Inf)
#}

# ============================================================
# SECTION 6: Temporal shift — does ministry targeting change across Lok Sabhas?
# ============================================================
#{

pm_temporal <- matched_filtered %>%
  count(lok_no, party_family, ministry_clean) %>%
  group_by(lok_no, party_family) %>%
  mutate(share = n / sum(n)) %>%
  ungroup()

# Focus on BJP and INC across top 12 ministries by total count
top_ministries <- ministry_counts %>%
  slice_max(n, n = 12) %>%
  pull(ministry_clean)

p_temporal <- pm_temporal %>%
  filter(party_family %in% c("BJP", "INC"),
         ministry_clean %in% top_ministries) %>%
  mutate(lok_label = paste0(lok_no, "th LS"),
         party_label = party_family) %>%
  ggplot(aes(x = lok_label, y = share * 100,
             group = party_label, colour = party_label)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  facet_wrap(~ reorder(ministry_clean, -share), ncol = 4, scales = "free_y") +
  scale_colour_manual(values = c(BJP = "#FF6B35", INC = "#1F78B4"),
                      name = NULL) +
  labs(
    title    = "BJP vs INC: how ministry targeting has shifted across Lok Sabhas",
    subtitle = "Share of each party's starred questions directed at each ministry, 16th–18th LS",
    x = NULL, y = "Share of questions (%)",
    caption  = "Top 12 ministries by total starred question volume shown."
  ) +
  theme_minimal(base_size = 9.5) +
  theme(
    legend.position = "bottom",
    plot.title      = element_text(face = "bold"),
    strip.text      = element_text(face = "bold", size = 8.5)
  )

ggsave(file.path(FIGDIR, "ministry_temporal.png"),
       p_temporal, width = 12, height = 7, dpi = 180)
cat("Saved: ministry_temporal.png\n")
#}

# ============================================================
# SECTION 7: Bipartite network — party × ministry
# ============================================================
# Edge weight = Pearson residual (positive only = over-questioning).
# Shows which parties are distinctively linked to which ministries.
#{

# Use igraph if available, otherwise skip
if (requireNamespace("igraph", quietly = TRUE)) {
  library(igraph)

  # Build edge list from significant over-targeting (pearson_resid > 1.5)
  edges <- pm_excess %>%
    filter(party_family %in% major_parties,
           ministry_clean %in% top_ministries,
           pearson_resid > 1.5) %>%
    select(party_family, ministry_clean, pearson_resid)

  g <- graph_from_data_frame(
    edges %>% rename(from = party_family, to = ministry_clean, weight = pearson_resid),
    directed = FALSE
  )

  party_nodes    <- unique(edges$party_family)
  ministry_nodes <- unique(edges$ministry_clean)

  party_colors <- c(
    BJP="#FF6B35", INC="#1F78B4", Left="#E31A1C", BSP="#6A3D9A",
    AAP="#33A02C", TMC="#B15928", SP="#FF7F00", JDU="#A6CEE3",
    DMK="#B2DF8A", TDP="#FDBF6F", `Shiv Sena`="#FB9A99",
    RJD="#CAB2D6", NCP="#999999"
  )

  node_type  <- ifelse(V(g)$name %in% party_nodes, "party", "ministry")
  node_color <- ifelse(node_type == "party",
                       party_colors[V(g)$name],
                       "#E5E5E5")
  node_size  <- ifelse(node_type == "party", 18, 12)

  png(file.path(FIGDIR, "ministry_network.png"),
      width = 2400, height = 1800, res = 200)
  set.seed(42)
  plot(
    g,
    layout           = layout_with_fr(g),
    vertex.color     = node_color,
    vertex.size      = node_size,
    vertex.label     = V(g)$name,
    vertex.label.cex = ifelse(node_type == "party", 0.75, 0.55),
    vertex.label.color = ifelse(node_type == "party", "white", "#333333"),
    vertex.frame.color = NA,
    edge.width       = E(g)$weight * 0.5,
    edge.color       = adjustcolor("grey40", alpha.f = 0.5),
    main             = "Party-Ministry Over-Targeting Network"
  )
  legend("bottomleft",
         legend = c("Party node", "Ministry node"),
         fill   = c("#FF6B35", "#E5E5E5"),
         border = NA, bty = "n", cex = 0.75)
  dev.off()
  cat("Saved: ministry_network.png\n")
} else {
  cat("igraph not installed — skipping network plot\n")
  cat("Install with: install.packages('igraph')\n")
}
#}

cat("\n=== C1 complete ===\n")
cat("Ministry analysis figures saved to:", FIGDIR, "\n")
