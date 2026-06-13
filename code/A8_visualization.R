# =============================================================================
# A8_visualization.R — UMAP / t-SNE + Final Summary Figures
# Author: Piyush Zaware
# Last updated: 2026-06-12
#
# PURPOSE:
#   Final publication-quality figures:
#     1. UMAP of party × session documents (colored by party)
#     2. t-SNE of same
#     3. Topic evolution stacked area: BJP 16th → 17th → 18th LS
#     4. Ministry heatmap: which ministries do BJP vs INC question most?
#     5. Summary panel combining ideological space + UMAP
#
# INPUTS:  all outputs from A4–A7
# OUTPUTS: $OUTDIR/figures/umap_*.png, tsne_*.png, topic_stacked_*.png,
#           ministry_heatmap.png, final_summary_panel.png
# =============================================================================

library(tidyverse)
library(umap)
library(Rtsne)
library(ggrepel)
library(patchwork)

set.seed(42)

# ============================================================
# SECTION 1: Load
# ============================================================
#{

tfidf_wide  <- readRDS(file.path(TMPDIR, "tfidf_wide.rds"))
doc_meta    <- read_csv(file.path(TMPDIR, "doc_meta_party.csv"))
lda_gamma   <- read_csv(file.path(OUTDIR, "tables", "lda_party_topics.csv"))
party_dims  <- read_csv(file.path(OUTDIR, "tables", "party_dimensions.csv"))
questions   <- readRDS(file.path(INPDIR, "questions_with_party.rds"))

doc_ids   <- tfidf_wide$doc_party_session
tfidf_mat <- tfidf_wide %>% select(-doc_party_session) %>% as.matrix()
rownames(tfidf_mat) <- doc_ids

party_colors <- c(
  "BJP"="saddlebrown","INC"="darkgreen","Left"="red3",
  "BSP"="purple4","AAP"="royalblue","TMC"="cyan4",
  "SP"="orange3","JDU"="brown","Regional"="grey50","Unknown"="grey80"
)

meta_join <- doc_meta %>%
  filter(doc_party_session %in% doc_ids) %>%
  mutate(label = paste0(substr(party_family,1,5), " LS", lok_no, "S", session_no))

#}

# ============================================================
# SECTION 2: UMAP
# ============================================================
#{

cfg <- umap.defaults
cfg$n_neighbors  <- min(15, nrow(tfidf_mat)-1)
cfg$min_dist     <- 0.2
cfg$random_state <- 42

cat("Running UMAP...\n")
umap_res <- umap::umap(tfidf_mat, config = cfg)

umap_df <- tibble(doc_party_session = doc_ids,
                  U1 = umap_res$layout[,1],
                  U2 = umap_res$layout[,2]) %>%
  left_join(meta_join, by = "doc_party_session")

p_umap <- umap_df %>%
  ggplot(aes(U1, U2, color = party_family, label = label)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = party_colors, name = "Party") +
  labs(title = "UMAP: Lok Sabha Starred Questions by Party × Session",
       subtitle = "TF-IDF manifold; each point = one party in one session",
       x = "UMAP 1", y = "UMAP 2") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "right")

ggsave(file.path(OUTDIR, "figures", "umap_party_session.png"),
       p_umap, width = 11, height = 7, dpi = 300)

#}

# ============================================================
# SECTION 3: t-SNE
# ============================================================
#{

cat("Running t-SNE...\n")
pca50  <- prcomp(tfidf_mat, center=TRUE)$x[, 1:min(50, ncol(tfidf_mat))]
perp   <- min(30, floor((nrow(tfidf_mat)-1)/3))
tsne   <- Rtsne(pca50, dims=2, perplexity=perp, max_iter=2000,
                verbose=FALSE, pca=FALSE)

tsne_df <- tibble(doc_party_session = doc_ids,
                  T1 = tsne$Y[,1], T2 = tsne$Y[,2]) %>%
  left_join(meta_join, by = "doc_party_session")

p_tsne <- tsne_df %>%
  ggplot(aes(T1, T2, color = party_family)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = party_colors, name = "Party") +
  labs(title = "t-SNE: Lok Sabha Starred Questions by Party × Session",
       x = "t-SNE 1", y = "t-SNE 2") +
  theme_minimal(base_size = 12)

ggsave(file.path(OUTDIR, "figures", "tsne_party_session.png"),
       p_tsne, width = 11, height = 7, dpi = 300)

#}

# ============================================================
# SECTION 4: Topic evolution stacked area — BJP across Lok Sabhas
# ============================================================
#{

make_stacked <- function(party_pat, title_str) {
  lda_gamma %>%
    filter(grepl(party_pat, party_family)) %>%
    group_by(lok_no, topic) %>%
    summarise(mean_gamma = mean(gamma), .groups = "drop") %>%
    ggplot(aes(factor(lok_no), mean_gamma,
               fill = factor(topic), group = factor(topic))) +
    geom_area(stat = "identity", alpha = 0.85) +
    scale_fill_viridis_d(option = "turbo", name = "Topic") +
    labs(title = paste("Topic Mix:", title_str),
         x = "Lok Sabha", y = "Mean Topic Proportion") +
    theme_minimal(base_size = 11)
}

ggsave(file.path(OUTDIR, "figures", "topic_stacked_bjp.png"),
       make_stacked("BJP",  "BJP Questions (16th → 17th → 18th LS)"),
       width = 8, height = 5, dpi = 300)

ggsave(file.path(OUTDIR, "figures", "topic_stacked_inc.png"),
       make_stacked("INC",  "INC Questions (16th → 17th → 18th LS)"),
       width = 8, height = 5, dpi = 300)

#}

# ============================================================
# SECTION 5: Ministry heatmap (BJP vs INC vs Left)
# ============================================================
#{

top_ministries <- questions %>%
  filter(type == "STARRED",
         party_family %in% c("BJP","INC","Left","BSP","AAP","TMC")) %>%
  dplyr::count(party_family, ministry, sort = TRUE) %>%
  group_by(party_family) %>%
  slice_max(n, n = 15) %>%
  ungroup() %>%
  group_by(ministry) %>%
  filter(n() >= 2) %>%   # keep ministries questioned by ≥2 parties
  ungroup()

ministry_wide <- top_ministries %>%
  pivot_wider(names_from = party_family, values_from = n, values_fill = 0) %>%
  mutate(across(-ministry, ~. / sum(.) ))  # normalize by party

p_ministry <- top_ministries %>%
  group_by(party_family) %>%
  mutate(share = n / sum(n)) %>%
  ggplot(aes(party_family, reorder(ministry, n), fill = share)) +
  geom_tile(color = "white") +
  scale_fill_distiller(palette = "YlOrRd", direction = 1, name = "Share") +
  labs(title = "Ministry Focus: Which Ministries Do Parties Question?",
       subtitle = "Share of starred questions directed at each ministry",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 30, hjust=1))

ggsave(file.path(OUTDIR, "figures", "ministry_heatmap.png"),
       p_ministry, width = 10, height = 10, dpi = 300)

#}

# ============================================================
# SECTION 6: Final summary panel
# ============================================================
#{

p_ideology <- party_dims %>%
  group_by(party_family) %>%
  summarise(hindutva=mean(pos_hindutva), econ_l=mean(pos_econ_l)) %>%
  ggplot(aes(econ_l, hindutva, label = party_family)) +
  geom_point(aes(color = party_family), size = 4, show.legend = FALSE) +
  scale_color_manual(values = party_colors) +
  geom_text_repel(size = 3) +
  geom_hline(yintercept=0, linetype="dashed", color="grey70") +
  geom_vline(xintercept=0, linetype="dashed", color="grey70") +
  labs(x="Economic Left ← → Right", y="Secular ← → Nationalist") +
  theme_minimal(base_size=10)

panel <- (p_ideology | p_umap) +
  plot_annotation(
    title    = "Indian Parliament: Unsupervised ML Analysis of Starred Questions",
    subtitle = "Left: ideological space (word2vec dims) | Right: UMAP manifold (TF-IDF)",
    theme    = theme(plot.title = element_text(size=14, face="bold"))
  )

ggsave(file.path(OUTDIR, "figures", "final_summary_panel.png"),
       panel, width = 18, height = 7, dpi = 300)

cat("\nA8 complete. All outputs in", OUTDIR, "\n")

#}
