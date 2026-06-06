library(ggplot2)
library(dplyr)
library(sf)
library(showtext)
showtext_auto()

# 古文嵌入tsne降维图

tsne_co <- read.csv("data/LTOitem_embeddings/lto_qwen3_tsne.csv")

ggplot(tsne_co, aes(x, y, fill = items, shape = is_guwen)) +
  geom_point(alpha = 0.7, stroke = 0.2, size = 3) +
  stat_ellipse(aes(group = items), col = "darkgrey", fill = NA, geom = "polygon", linetype = 2, size = 0.5, alpha = 0.2) +
  scale_fill_manual(values = c(
    "帮助朋友，慷慨大方。" = "#d62728",
    "节俭（除必要花销外，不多花钱）" = "#bcbd22",
    "作为中国公民，您感到自豪吗？" = "#9467bd",
    "持续的努力是取得好结果最可靠的方式" = "#1f77b4"
  )) +
  scale_shape_manual(values = c(24, 21)) +
  theme_bw() +
  labs(x = "t-SNE Dimension 1", y = "t-SNE Dimension 2") +
  theme(
    legend.position = c(0.02, 0.98),
    legend.justification = c(0, 1),
    legend.title = element_blank(),
    legend.background = element_rect(fill = alpha("white", 0.6), color = NA)
  ) +
  guides(
    fill = guide_legend(order = 1, override.aes = list(shape = 21, size = 3)),
    shape = guide_legend(order = 2, override.aes = list(fill = "black", size = 3))
  )

tsne_co_ancient <- read.csv("data/LTOitem_embeddings/lto_ancient_tsne.csv")

ggplot(tsne_co_ancient, aes(x, y, fill = items)) +
  geom_point(alpha = 0.7, stroke = 0.2, size = 3) +
  stat_ellipse(aes(group = items), col = "darkgrey", fill = NA, geom = "polygon", linetype = 2, size = 0.5, alpha = 0.2) +
  scale_fill_manual(values = c(
    "帮助朋友，慷慨大方。" = "#d62728",
    "节俭（除必要花销外，不多花钱）" = "#bcbd22",
    "作为中国公民，您感到自豪吗？" = "#9467bd",
    "持续的努力是取得好结果最可靠的方式" = "#1f77b4"
  )) +
  scale_shape_manual(values = c(24, 21)) +
  theme_bw() +
  labs(x = "t-SNE Dimension 1", y = "t-SNE Dimension 2") +
  theme(
    legend.position = c(0.02, 0.98),
    legend.justification = c(0, 1),
    legend.title = element_blank(),
    legend.background = element_rect(fill = alpha("white", 0.6), color = NA)
  ) +
  guides(
    fill = guide_legend(order = 1, override.aes = list(shape = 21, size = 3)),
    shape = guide_legend(order = 2, override.aes = list(fill = "black", size = 3))
  )

# LTO 地图

lto <- read.csv("data/LTO_fu_scores.csv")

TARGET_YR <- 1393

hist_pref <- st_read(
  "data/chgis地图多边形/v6_time_pref_pgn_utf_wgs84/v6_time_pref_pgn_utf_wgs84.shp",
  quiet = TRUE
) %>%
  st_transform(4326) %>%
  filter(BEG_YR < TARGET_YR & END_YR >= TARGET_YR) %>%
  st_make_valid()

map_data <- left_join(hist_pref, lto, by = "NAME_CH")

ggplot(map_data) +
  geom_sf(aes(fill = LTO), color = "darkgrey") +
  scale_fill_gradient(high = "#2f7fc1", low = "white", name = "LTO") +
  theme_bw() +
  theme(legend.position = "top")
