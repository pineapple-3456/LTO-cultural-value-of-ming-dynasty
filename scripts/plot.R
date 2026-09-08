library(ggplot2)
library(dplyr)
library(sf)
library(showtext)
showtext_auto()

# 图 1：全部 44 个题项。CSV 由 scripts/tsne_items.py 根据已保存向量生成，
# 本脚本只负责用 R 绘图。
tsne_co <- read.csv("data/LTOitem_embeddings/lto_qwen3_tsne.csv")

p_all <- ggplot(tsne_co, aes(x, y, fill = items, shape = is_guwen)) +
  geom_point(alpha = 0.7, stroke = 0.2, size = 3) +
  stat_ellipse(aes(x = x, y = y, group = items), inherit.aes = FALSE, col = "darkgrey", fill = NA, geom = "polygon", linetype = 2, linewidth = 0.5, alpha = 0.2) +
  scale_fill_manual(limits = c(
    "帮助朋友，慷慨大方。",
    "持续的努力是取得好结果最可靠的方式",
    "节俭（除必要花销外，不多花钱）",
    "作为中国公民，您感到自豪吗？"
  ), values = c(
    "帮助朋友，慷慨大方。" = "#d62728",
    "节俭（除必要花销外，不多花钱）" = "#bcbd22",
    "作为中国公民，您感到自豪吗？" = "#9467bd",
    "持续的努力是取得好结果最可靠的方式" = "#1f77b4"
  )) +
  scale_shape_manual(values = c(24, 21), labels = c("题项原文", "古文句子")) +
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
ggsave("output/tsne_all.pdf", p_all, width = 8.25, height = 6.611111, units = "in")

# 图 2：40 条古文比较句子，沿用图 1 的题项颜色。
tsne_co_ancient <- read.csv("data/LTOitem_embeddings/lto_ancient_tsne.csv")

p_ancient <- ggplot(tsne_co_ancient, aes(x, y, fill = items)) +
  geom_point(shape = 21, alpha = 0.7, stroke = 0.2, size = 3) +
  stat_ellipse(aes(x = x, y = y, group = items), inherit.aes = FALSE, col = "darkgrey", fill = NA, geom = "polygon", linetype = 2, linewidth = 0.5, alpha = 0.2) +
  scale_fill_manual(limits = c(
    "帮助朋友，慷慨大方。",
    "持续的努力是取得好结果最可靠的方式",
    "节俭（除必要花销外，不多花钱）",
    "作为中国公民，您感到自豪吗？"
  ), values = c(
    "帮助朋友，慷慨大方。" = "#d62728",
    "节俭（除必要花销外，不多花钱）" = "#bcbd22",
    "作为中国公民，您感到自豪吗？" = "#9467bd",
    "持续的努力是取得好结果最可靠的方式" = "#1f77b4"
  )) +
  theme_bw() +
  labs(x = "t-SNE Dimension 1", y = "t-SNE Dimension 2") +
  theme(
    legend.position = c(0.02, 0.98),
    legend.justification = c(0, 1),
    legend.title = element_blank(),
    legend.background = element_rect(fill = alpha("white", 0.6), color = NA)
  ) +
  guides(fill = guide_legend(order = 1, override.aes = list(shape = 21, size = 3)))
ggsave("output/tsne_ancient.pdf", p_ancient, width = 8.25, height = 6.611111, units = "in")

# 图 3：按稳定的 SYS_ID 将分数连接到 CHGIS 多边形。年份筛选只保留目标年
# 有效的多边形；没有分数的单位（例如东番）保留 NA，由 ggplot 用缺失值颜色绘制。
lto <- read.csv("data/LTO_fu_scores.csv")
TARGET_YR <- 1393

hist_pref <- st_read(
  "data/chgis地图多边形/v6_time_pref_pgn_utf_wgs84/v6_time_pref_pgn_utf_wgs84.shp",
  quiet = TRUE
) %>%
  st_transform(4326) %>%
  filter(BEG_YR <= TARGET_YR, END_YR >= TARGET_YR) %>%
  st_make_valid()

map_data <- left_join(hist_pref, lto %>% select(SYS_ID, LTO), by = "SYS_ID")

p_map <- ggplot(map_data) +
  geom_sf(aes(fill = LTO), color = "darkgrey") +
  scale_fill_gradient(high = "#2f7fc1", low = "white", name = "LTO") +
  theme_bw() +
  theme(legend.position = "top")
ggsave("output/lto_map.pdf", p_map, width = 7.56, height = 7.39, units = "in")
