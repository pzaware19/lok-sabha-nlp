# =============================================================================
# D1_conetwork.R — MP Co-Questioning Network Analysis
# Author: Piyush Zaware
# Updated: 2026-06-13
#
# PURPOSE:
#   Builds an MP × MP network where an edge connects two MPs who co-signed
#   the same starred question. Edge weight = number of questions co-signed.
#   Node attributes: party, Lok Sabha, state/constituency.
#
#   Research questions:
#   (1) Is co-questioning mostly within-party or do MPs coordinate across
#       party lines?
#   (2) Which MPs are the most central connectors (high betweenness)?
#   (3) Are there regional clusters that cut across party lines?
#   (4) Has cross-party coordination changed across 16th-18th Lok Sabha?
#   (5) What is the party-level co-questioning matrix? (party × party heatmap)
#
#   Note on Indian parliamentary practice:
#   Starred questions can list a primary questioner + supplementary signatories.
#   ~39% of starred questions have 2+ co-signers. Co-signing is strategic:
#   parties coordinate to ensure coverage, and regional MPs co-sign with
#   national parties to amplify constituency issues.
#
# INPUTS:
#   tmp/train-*.parquet           — raw questions
#   input/mp_party_lookup.csv     — MP → party_family (1,642 MPs)
#
# OUTPUTS:
#   output/figures/conet_full.png          — full MP network, party-colored
#   output/figures/conet_cross_party.png   — cross-party edges only
#   output/figures/conet_party_matrix.png  — party × party co-questioning heatmap
#   output/figures/conet_top_connectors.png — top MPs by betweenness centrality
#   output/figures/conet_temporal.png      — cross-party rate by Lok Sabha
#   output/tables/conet_edges.csv          — edge list with party info
#   output/tables/conet_nodes.csv          — node list with centrality measures
#   output/tables/conet_party_matrix.csv   — party × party matrix
# =============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(tidyverse)
  library(igraph)
  library(ggraph)
  library(patchwork)
  library(ggrepel)
})

if (!exists("OUTDIR")) {
  root   <- "/Users/piyushzaware/Documents/Unsupervised ML/Lok_Sabha_Questions"
  INPDIR <- file.path(root, "input")
  CODDIR <- file.path(root, "code")
  OUTDIR <- file.path(root, "output")
  TMPDIR <- file.path(root, "tmp")
}

FIGDIR <- file.path(OUTDIR, "figures")
TABDIR <- file.path(OUTDIR, "tables")

# ============================================================
# SECTION 1: Load data and extract co-signatory pairs
# ============================================================
#{

cat("Loading parquet files...\n")
parquet_files <- list.files(TMPDIR, pattern = "train-.*\\.parquet$", full.names = TRUE)

raw <- purrr::map_dfr(parquet_files, function(f) {
  read_parquet(f, col_select = c("id", "lok_no", "session_no", "type", "members"))
})

starred <- raw %>% filter(type == "STARRED")
cat("  Starred questions:", nrow(starred), "\n")

# Extract member list from Arrow list column
starred <- starred %>%
  mutate(
    member_list = purrr::map(members, function(x) {
      items <- tryCatch(as.character(x), error = function(e) character(0))
      items <- str_squish(items)
      items[nchar(items) > 2]
    }),
    n_members = lengths(member_list)
  )

cat("  Questions with 2+ co-signatories:",
    sum(starred$n_members >= 2), "\n")
cat("  Questions with 1 signatory:", sum(starred$n_members == 1), "\n")

# Keep only questions with 2+ members — these generate edges
multi <- starred %>% filter(n_members >= 2)
#}

# ============================================================
# SECTION 2: Generate all MP pairs per question
# ============================================================
#{

# For each question, generate all (MP_i, MP_j) pairs (i < j)
# Edge weight = number of questions the pair co-signed together
cat("Generating MP pairs...\n")

edge_rows <- multi %>%
  purrr::pmap_dfr(function(id, lok_no, session_no, member_list, ...) {
    members_clean <- str_to_upper(str_squish(member_list))
    if (length(members_clean) < 2) return(NULL)
    # All unique pairs
    pairs <- combn(members_clean, 2, simplify = FALSE)
    purrr::map_dfr(pairs, function(p) {
      tibble(mp1 = p[1], mp2 = p[2], question_id = id,
             lok_no = lok_no, session_no = session_no)
    })
  })

cat("  Total co-signatory pairs:", nrow(edge_rows), "\n")
cat("  Unique MP pairs:", n_distinct(paste(edge_rows$mp1, edge_rows$mp2)), "\n")
#}

# ============================================================
# SECTION 3: Party matching
# ============================================================
#{

crosswalk <- read_csv(file.path(INPDIR, "mp_name_crosswalk.csv"),
                      show_col_types = FALSE) %>%
  mutate(raw_upper = str_to_upper(raw_name)) %>%
  arrange(desc(lok_no)) %>%
  distinct(raw_upper, .keep_all = TRUE)

lookup_const <- read_csv(file.path(INPDIR, "mp_party_lookup.csv"),
                         show_col_types = FALSE) %>%
  distinct(mp_name, constituency) %>%
  deframe()

mp_to_party        <- setNames(crosswalk$party_family, crosswalk$raw_upper)
mp_to_constituency <- setNames(
  lookup_const[crosswalk$matched_mp_name], crosswalk$raw_upper)

edges <- edge_rows %>%
  mutate(
    party1 = mp_to_party[mp1],
    party2 = mp_to_party[mp2]
  )

# Cross-party edge = both MPs matched AND different parties
edges <- edges %>%
  mutate(
    matched     = !is.na(party1) & !is.na(party2),
    cross_party = matched & (party1 != party2)
  )

cat("  Pairs with both MPs matched:", sum(edges$matched), "\n")
cat("  Cross-party pairs:",
    sum(edges$cross_party, na.rm = TRUE),
    sprintf("(%.1f%% of matched)\n",
            100 * mean(edges$cross_party, na.rm = TRUE)))
#}

# ============================================================
# SECTION 4: Build edge list (aggregated by MP pair)
# ============================================================
#{

edge_list <- edges %>%
  filter(matched) %>%
  group_by(mp1, mp2, party1, party2) %>%
  summarise(
    weight      = n(),
    cross_party = first(party1) != first(party2),
    lok_nos     = paste(sort(unique(lok_no)), collapse = ","),
    .groups     = "drop"
  ) %>%
  arrange(desc(weight))

write_csv(edge_list, file.path(TABDIR, "conet_edges.csv"))
cat("Saved: conet_edges.csv\n")
cat("  Unique MP pairs (matched):", nrow(edge_list), "\n")
cat("  Cross-party pairs:", sum(edge_list$cross_party), "\n")
#}

# ============================================================
# SECTION 5: Build igraph object + node attributes
# ============================================================
#{

# Build graph from edge list
g <- graph_from_data_frame(
  d        = edge_list %>% select(from = mp1, to = mp2, weight, cross_party),
  directed = FALSE
)

# Node attributes
all_mps <- unique(c(edge_list$mp1, edge_list$mp2))
node_party       <- mp_to_party[all_mps]
node_constituency <- mp_to_constituency[all_mps]

V(g)$party        <- node_party[V(g)$name]
V(g)$constituency <- node_constituency[V(g)$name]
V(g)$degree       <- degree(g, mode = "all")
V(g)$strength     <- strength(g, weights = E(g)$weight)  # weighted degree

# Betweenness on largest connected component
# (betweenness is undefined / trivially 0 on disconnected graph)
comps     <- components(g)
lcc_ids   <- which(comps$membership == which.max(comps$csize))
g_lcc     <- induced_subgraph(g, lcc_ids)
btw_lcc   <- betweenness(g_lcc, normalized = TRUE)
V(g)$betweenness <- 0
V(g)$betweenness[lcc_ids] <- btw_lcc

node_df <- tibble(
  mp            = V(g)$name,
  party         = V(g)$party,
  constituency  = V(g)$constituency,
  degree        = V(g)$degree,
  strength      = V(g)$strength,
  betweenness   = V(g)$betweenness
) %>% arrange(desc(betweenness))

write_csv(node_df, file.path(TABDIR, "conet_nodes.csv"))
cat("Saved: conet_nodes.csv\n")
cat("  Network: ", vcount(g), "nodes,", ecount(g), "edges\n")
cat("  LCC size:", vcount(g_lcc), "nodes\n")
cat("  Graph density:", round(graph.density(g), 4), "\n")
#}

# ============================================================
# SECTION 6: Party × party co-questioning matrix
# ============================================================
#{

PARTY_COLORS <- c(
  BJP        = "#FF6B35", INC        = "#1F78B4",
  Left       = "#E31A1C", BSP        = "#6A3D9A",
  AAP        = "#33A02C", TMC        = "#B15928",
  SP         = "#FF7F00", JDU        = "#A6CEE3",
  DMK        = "#B2DF8A", TDP        = "#FDBF6F",
  `Shiv Sena`= "#FB9A99", RJD        = "#CAB2D6",
  NCP        = "#999999", Regional   = "#CCCCCC"
)

party_matrix <- edge_list %>%
  count(party1, party2, wt = weight, name = "n_coquestions") %>%
  # Symmetrize: (A,B) and (B,A) → same cell
  mutate(
    p_lo = pmin(party1, party2),
    p_hi = pmax(party1, party2)
  ) %>%
  group_by(p_lo, p_hi) %>%
  summarise(n_coquestions = sum(n_coquestions), .groups = "drop")

# Self-pairs (within party) vs cross-party
party_matrix_sym <- bind_rows(
  party_matrix %>% rename(party1 = p_lo, party2 = p_hi),
  party_matrix %>% filter(p_lo != p_hi) %>%
    rename(party1 = p_hi, party2 = p_lo)
)

write_csv(party_matrix, file.path(TABDIR, "conet_party_matrix.csv"))
cat("Saved: conet_party_matrix.csv\n")

major_parties <- c("BJP", "INC", "Left", "JDU", "DMK", "TDP",
                   "Shiv Sena", "Regional", "BSP", "SP", "RJD")

p_party_matrix <- party_matrix_sym %>%
  filter(party1 %in% major_parties, party2 %in% major_parties) %>%
  mutate(
    is_self = party1 == party2,
    log_n   = log1p(n_coquestions)
  ) %>%
  ggplot(aes(x = party1, y = party2, fill = log_n)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = n_coquestions), size = 2.8, colour = "grey20") +
  scale_fill_distiller(palette = "YlOrRd", direction = 1,
                       name = "log(n+1)\nco-questions") +
  labs(
    title    = "Party × party co-questioning matrix",
    subtitle = "Number of starred question pairs co-signed by one MP from each party.\nDiagonal = within-party co-signing.",
    x = NULL, y = NULL,
    caption  = "16th–18th Lok Sabha. Only matched pairs shown."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x  = element_text(angle = 35, hjust = 1),
    plot.title   = element_text(face = "bold"),
    legend.position = "right"
  )

ggsave(file.path(FIGDIR, "conet_party_matrix.png"),
       p_party_matrix, width = 10, height = 8, dpi = 180)
cat("Saved: conet_party_matrix.png\n")
#}

# ============================================================
# SECTION 7: Full network — ggraph visualization
# ============================================================
#{

# Focus on nodes with degree >= 2 to reduce clutter
g_plot <- induced_subgraph(g, V(g)[V(g)$degree >= 2 & !is.na(V(g)$party)])

cat("Plotting network (nodes with degree ≥ 2):", vcount(g_plot), "nodes\n")

# Convert to tidygraph for ggraph
library(tidygraph)
tg <- as_tbl_graph(g_plot) %>%
  activate(nodes) %>%
  mutate(
    party      = V(g_plot)$party,
    betweenness = V(g_plot)$betweenness,
    strength   = V(g_plot)$strength,
    label      = if_else(betweenness > quantile(betweenness, 0.97, na.rm = TRUE),
                         str_to_title(name), NA_character_)
  ) %>%
  activate(edges) %>%
  mutate(cross_party = cross_party)

set.seed(42)
p_full <- ggraph(tg, layout = "fr") +
  geom_edge_link(aes(alpha = weight, colour = cross_party),
                 linewidth = 0.3, show.legend = TRUE) +
  scale_edge_colour_manual(
    values = c(`FALSE` = "grey80", `TRUE` = "#B5440E"),
    labels = c("Within-party", "Cross-party"),
    name   = "Edge type"
  ) +
  scale_edge_alpha(range = c(0.05, 0.6), guide = "none") +
  geom_node_point(aes(colour = party, size = strength),
                  alpha = 0.85) +
  geom_node_text(aes(label = str_to_title(label)),
                 size = 2.2, repel = TRUE, max.overlaps = 20,
                 colour = "grey20", bg.colour = "white", bg.r = 0.1) +
  scale_colour_manual(values = PARTY_COLORS, na.value = "#AAAAAA",
                      name = "Party") +
  scale_size_continuous(range = c(1.5, 7), guide = "none") +
  labs(
    title    = "MP co-questioning network",
    subtitle = "Edge = two MPs co-signed the same starred question. Orange edges = cross-party.\nNode size = weighted degree (total co-signed questions). Labels = top 3% by betweenness.",
    caption  = "Nodes with degree ≥ 2 shown. Layout: Fruchterman-Reingold."
  ) +
  theme_graph(base_family = "sans") +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 9),
    legend.position = "right"
  )

ggsave(file.path(FIGDIR, "conet_full.png"),
       p_full, width = 14, height = 11, dpi = 180)
cat("Saved: conet_full.png\n")
#}

# ============================================================
# SECTION 8: Cross-party edges only — who bridges parties?
# ============================================================
#{

cross_edges <- edge_list %>%
  filter(cross_party) %>%
  arrange(desc(weight))

cat("\nTop 20 cross-party co-questioning pairs:\n")
print(cross_edges %>%
        select(mp1, party1, mp2, party2, weight) %>%
        head(20), n = 20)

# Network of cross-party edges only
g_cross <- graph_from_data_frame(
  cross_edges %>%
    filter(!is.na(party1), !is.na(party2)) %>%
    select(from = mp1, to = mp2, weight, party1, party2),
  directed = FALSE
)
V(g_cross)$party <- mp_to_party[V(g_cross)$name]

tg_cross <- as_tbl_graph(g_cross) %>%
  activate(nodes) %>%
  mutate(
    party    = V(g_cross)$party,
    deg      = degree(g_cross),
    label    = if_else(deg >= 3, str_to_title(name), NA_character_)
  )

set.seed(7)
p_cross <- ggraph(tg_cross, layout = "fr") +
  geom_edge_link(aes(alpha = weight, width = weight),
                 colour = "#B5440E", show.legend = FALSE) +
  scale_edge_alpha(range = c(0.2, 0.8), guide = "none") +
  scale_edge_width(range = c(0.3, 2.5), guide = "none") +
  geom_node_point(aes(colour = party, size = deg), alpha = 0.9) +
  geom_node_text(aes(label = str_to_title(label)),
                 size = 2.5, repel = TRUE, max.overlaps = 30,
                 colour = "grey20", bg.colour = "white", bg.r = 0.1) +
  scale_colour_manual(values = PARTY_COLORS, na.value = "#AAAAAA",
                      name = "Party") +
  scale_size_continuous(range = c(2, 9), guide = "none") +
  labs(
    title    = "Cross-party co-questioning network",
    subtitle = "Only edges between MPs of different parties shown.\nEdge width = number of questions co-signed. Labels = MPs with 3+ cross-party partners.",
    caption  = "Cross-party coordination reveals coalition politics beyond formal alliances."
  ) +
  theme_graph(base_family = "sans") +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 9),
    legend.position = "right"
  )

ggsave(file.path(FIGDIR, "conet_cross_party.png"),
       p_cross, width = 13, height = 10, dpi = 180)
cat("Saved: conet_cross_party.png\n")
#}

# ============================================================
# SECTION 9: Top MPs by betweenness centrality
# ============================================================
#{

top_connectors <- node_df %>%
  filter(!is.na(party), betweenness > 0) %>%
  slice_max(betweenness, n = 25) %>%
  mutate(
    name_clean = str_to_title(mp),
    party      = factor(party, levels = names(PARTY_COLORS))
  )

p_btw <- ggplot(top_connectors,
                aes(x = reorder(name_clean, betweenness),
                    y = betweenness * 100,
                    fill = party)) +
  geom_col(width = 0.75) +
  geom_text(aes(label = paste0(party, " · ", str_to_title(constituency))),
            hjust = -0.05, size = 2.6, colour = "grey30") +
  coord_flip() +
  scale_fill_manual(values = PARTY_COLORS, na.value = "#AAAAAA",
                    guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.35)),
                     labels = scales::number_format(accuracy = 0.01)) +
  labs(
    title    = "Top 25 MPs by betweenness centrality in co-questioning network",
    subtitle = "Betweenness = fraction of shortest paths between all MP pairs passing through this node.\nHigh betweenness = bridge between otherwise disconnected groups.",
    x = NULL, y = "Betweenness centrality (×100)",
    caption  = "Computed on the largest connected component."
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title  = element_text(face = "bold"),
    axis.text.y = element_text(size = 8.5)
  )

ggsave(file.path(FIGDIR, "conet_top_connectors.png"),
       p_btw, width = 12, height = 9, dpi = 180)
cat("Saved: conet_top_connectors.png\n")
#}

# ============================================================
# SECTION 10: Cross-party rate by Lok Sabha
# ============================================================
#{

temporal_cross <- edges %>%
  filter(matched) %>%
  group_by(lok_no) %>%
  summarise(
    total_pairs   = n(),
    cross_pairs   = sum(cross_party),
    cross_pct     = 100 * mean(cross_party),
    .groups       = "drop"
  )

cat("\nCross-party co-questioning rate by Lok Sabha:\n")
print(temporal_cross)

p_temporal <- ggplot(temporal_cross,
                     aes(x = factor(lok_no), y = cross_pct, group = 1)) +
  geom_line(linewidth = 1.2, colour = "#B5440E") +
  geom_point(aes(size = total_pairs), colour = "#B5440E") +
  geom_text(aes(label = paste0(round(cross_pct, 1), "%\n(n=", total_pairs, ")")),
            vjust = -1, size = 3.2) +
  scale_size_continuous(range = c(4, 9), name = "Total pairs") +
  scale_y_continuous(limits = c(0, max(temporal_cross$cross_pct) * 1.3),
                     labels = scales::percent_format(scale = 1)) +
  labs(
    title    = "Cross-party co-questioning rate by Lok Sabha",
    subtitle = "% of matched co-signatory pairs from different parties.\nDecline suggests hardening of partisan boundaries over time.",
    x = "Lok Sabha", y = "% cross-party pairs",
    caption  = "16 = 2014–2019, 17 = 2019–2024, 18 = 2024–present."
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(FIGDIR, "conet_temporal.png"),
       p_temporal, width = 8, height = 5, dpi = 180)
cat("Saved: conet_temporal.png\n")
#}

# ============================================================
# SECTION 11: Community detection
# ============================================================
#{

# Louvain community detection on LCC
set.seed(42)
comm <- cluster_louvain(g_lcc, weights = E(g_lcc)$weight)
cat("\nLouvain community detection (LCC):\n")
cat("  Communities found:", length(comm), "\n")
cat("  Modularity:", round(modularity(comm), 3), "\n")

# Community composition: what fraction of each community is each party?
comm_df <- tibble(
  mp        = V(g_lcc)$name,
  community = comm$membership,
  party     = V(g_lcc)$party
) %>%
  filter(!is.na(party))

comm_summary <- comm_df %>%
  count(community, party) %>%
  group_by(community) %>%
  mutate(
    share     = n / sum(n),
    comm_size = sum(n)
  ) %>%
  ungroup() %>%
  filter(comm_size >= 5)  # skip tiny communities

cat("\nLargest communities and their party composition:\n")
comm_summary %>%
  slice_max(share, n = 1, by = community) %>%
  arrange(desc(comm_size)) %>%
  select(community, comm_size, dominant_party = party, share) %>%
  print(n = 15)

# Party purity: how many communities are >80% one party?
purity <- comm_summary %>%
  slice_max(share, n = 1, by = community) %>%
  summarise(
    n_pure      = sum(share > 0.80),
    n_mixed     = sum(share <= 0.80 & share > 0.50),
    total_comms = n()
  )
cat("\nCommunity purity:\n")
print(purity)
#}

cat("\n=== D1 complete ===\n")
cat("Outputs saved to:", FIGDIR, "\n")
