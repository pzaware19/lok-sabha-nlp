# =============================================================================
# A7_clustering.R — HAC + GMM Clustering
# Author: Piyush Zaware
# Last updated: 2026-06-12
#
# PURPOSE:
#   Cluster parties (aggregated across sessions) by their question text.
#   Key question: do parties cluster by ideology (BJP/right vs. INC/center
#   vs. Left) or by region (Hindi belt vs. South vs. East)?
#   Methods: HAC on cosine distance, GMM on PCA scores.
#
# INPUTS:  $TMPDIR/tfidf_wide.rds, doc_meta_party.csv, party_dimensions.csv
# OUTPUTS:
#   $OUTDIR/figures/hac_dendrogram.png
#   $OUTDIR/figures/gmm_pca_clusters.png
#   $OUTDIR/tables/cluster_assignments.csv
# =============================================================================

library(tidyverse)
library(mclust)
library(dendextend)
library(ggrepel)

set.seed(42)

# ============================================================
# SECTION 1: Load
# ============================================================
#{

tfidf_wide <- readRDS(file.path(TMPDIR, "tfidf_wide.rds"))
doc_meta   <- read_csv(file.path(TMPDIR, "doc_meta_party.csv"))
party_dims <- read_csv(file.path(OUTDIR, "tables", "party_dimensions.csv"))

doc_ids   <- tfidf_wide$doc_party_session
tfidf_mat <- tfidf_wide %>% select(-doc_party_session) %>% as.matrix()
rownames(tfidf_mat) <- doc_ids

# Aggregate to party level (mean TF-IDF across sessions) for cleaner clustering
doc_meta_join <- doc_meta %>%
  filter(doc_party_session %in% doc_ids)

# Build party-level matrix
party_ids <- doc_meta_join %>%
  group_by(party_family) %>%
  summarise(docs = list(doc_party_session), .groups = "drop")

party_mat <- purrr::map_dfr(party_ids$party_family, function(p) {
  docs <- doc_meta_join$doc_party_session[doc_meta_join$party_family == p]
  docs <- docs[docs %in% rownames(tfidf_mat)]
  if (length(docs) == 0) return(NULL)
  as_tibble(t(colMeans(tfidf_mat[docs, , drop=FALSE]))) %>%
    mutate(party_family = p)
}) %>%
  select(party_family, everything())

p_names <- party_mat$party_family
mat     <- party_mat %>% select(-party_family) %>% as.matrix()
rownames(mat) <- p_names
cat("Party matrix:", dim(mat), "\n")

#}

# ============================================================
# SECTION 2: HAC on cosine distance
# ============================================================
#{

mat_norm <- mat / sqrt(rowSums(mat^2) + 1e-10)
cos_dist <- as.dist(1 - tcrossprod(mat_norm))

hac <- hclust(cos_dist, method = "ward.D2")

party_colors <- c(
  "BJP"="saddlebrown", "INC"="darkgreen", "Left"="red3",
  "BSP"="purple4",     "AAP"="royalblue", "TMC"="cyan4",
  "SP"="orange3",      "JDU"="brown",     "DMK"="navy",
  "AIADMK"="goldenrod","TDP"="steelblue", "Regional"="grey50"
)
col_vec <- party_colors[hac$labels]
col_vec[is.na(col_vec)] <- "grey50"

png(file.path(OUTDIR, "figures", "hac_dendrogram.png"),
    width = 1200, height = 600, res = 130)
dend <- as.dendrogram(hac) %>%
  color_labels(col = col_vec[order.dendrogram(as.dendrogram(hac))])
plot(dend,
     main = "HAC: Indian Parties Clustered by Parliamentary Question Text",
     sub  = "Ward's D² on cosine distance of TF-IDF vectors",
     ylab = "Distance", cex = 0.85)
dev.off()

#}

# ============================================================
# SECTION 3: PCA + GMM
# ============================================================
#{

pca    <- prcomp(mat, center = TRUE, scale. = FALSE)
var_ex <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
n_pcs  <- min(which(var_ex >= 0.80), ncol(pca$x))
cat("PCs for 80% variance:", n_pcs, "\n")

scores <- pca$x[, 1:min(n_pcs, 10)]
gmm    <- Mclust(scores, G = 2:min(6, nrow(scores)-1))
cat("GMM selected:", gmm$G, "clusters, model:", gmm$modelName, "\n")

cluster_df <- tibble(
  party_family = rownames(mat),
  gmm_cluster  = gmm$classification,
  PC1          = pca$x[,1],
  PC2          = pca$x[,2]
)

p_pca <- cluster_df %>%
  ggplot(aes(PC1, PC2, color = factor(gmm_cluster), label = party_family)) +
  geom_point(size = 4) +
  geom_text_repel(size = 3.5) +
  scale_color_brewer(palette = "Set1", name = "Cluster") +
  labs(
    title = "GMM Clustering of Indian Parties (PCA space)",
    subtitle = paste0(gmm$G, " clusters selected by BIC; TF-IDF of starred questions"),
    x = paste0("PC1 (", round(var_ex[1]*100,1), "%)"),
    y = paste0("PC2 (", round((var_ex[2]-var_ex[1])*100,1), "%)")
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(OUTDIR, "figures", "gmm_pca_clusters.png"),
       p_pca, width = 9, height = 7, dpi = 300)

#}

# ============================================================
# SECTION 4: Clusters in embedding space
# ============================================================
#{

party_means <- party_dims %>%
  group_by(party_family) %>%
  summarise(hindutva = mean(pos_hindutva),
            econ_l   = mean(pos_econ_l), .groups = "drop") %>%
  left_join(cluster_df %>% select(party_family, gmm_cluster), by = "party_family")

p_embed <- party_means %>%
  ggplot(aes(econ_l, hindutva,
             color = factor(gmm_cluster), label = party_family)) +
  geom_point(size = 4) +
  geom_text_repel(size = 3.5) +
  scale_color_brewer(palette = "Set1", name = "Cluster") +
  geom_hline(yintercept = 0, linetype="dashed", color="grey70") +
  geom_vline(xintercept = 0, linetype="dashed", color="grey70") +
  labs(title = "Party Clusters in Ideological Space",
       x = "Economic Left ← → Right",
       y = "Secular ← → Nationalist") +
  theme_minimal(base_size = 13)

ggsave(file.path(OUTDIR, "figures", "embedding_clusters.png"),
       p_embed, width = 9, height = 7, dpi = 300)

write_csv(cluster_df, file.path(OUTDIR, "tables", "cluster_assignments.csv"))
cat("\nA7 complete.\n")

#}
