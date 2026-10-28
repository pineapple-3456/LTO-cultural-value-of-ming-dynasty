library(ggplot2)
library(dplyr)
library(sf)
library(showtext)
# showtext 负责 PDF 中的中文显示。
showtext_auto()

# Rscript 从 --file 参数取得脚本路径，source() 从调用信息取得路径。
# 脚本位于 scripts 目录，其上一级为项目根目录，下面的输入输出均以此为起点。
script <- if (sys.nframe() > 0) {
  sys.frame(1)$ofile
} else {
  sub("--file=", "", commandArgs()[grepl("^--file=", commandArgs())])
}
root <- dirname(dirname(normalizePath(script)))

plot_items <- function(data) {
  # 每个点代表一句话，颜色区分四个模块，形状区分题项原文和古文。
  # labels 将模块代码与中文题项对应，用于图例；两张图沿用同一组颜色。
  labels <- distinct(data, label, items)
  ggplot(data, aes(x, y, fill = label)) +
    geom_point(aes(shape = is_guwen), alpha = 0.7, stroke = 0.2, size = 3) +
    scale_fill_manual(values = c(m13 = "#d62728", m14 = "#bcbd22", m19 = "#9467bd", m22 = "#1f77b4"),
                      labels = setNames(labels$items, labels$label)) +
    scale_shape_manual(values = c("False" = 24, "True" = 21),
                       labels = c("题项原文", "古文句子")) +
    theme_bw() +
    labs(x = "t-SNE Dimension 1", y = "t-SNE Dimension 2") +
    theme(legend.position = "bottom", legend.title = element_blank()) +
    guides(fill = guide_legend(ncol = 1, override.aes = list(shape = 21)),
           shape = guide_legend(override.aes = list(fill = "black")))
}

# 图一使用 Qwen 向量，包含四条题项原文和 40 条古文；图二使用 BERT 向量，只包含 40 条古文。
# 古文图的所有点形状相同，因此只显示颜色图例。
all_items <- read.csv(file.path(root, "data/LTOitem_embeddings/lto_all_tsne.csv"))
ancient <- read.csv(file.path(root, "data/LTOitem_embeddings/lto_ancient_tsne.csv"))
p_all <- plot_items(all_items)
p_ancient <- plot_items(ancient) + guides(shape = "none")
ggsave(file.path(root, "output/tsne_all.pdf"), p_all, width = 8.25, height = 8)
ggsave(file.path(root, "output/tsne_ancient.pdf"), p_ancient, width = 8.25, height = 8)

# 图三将 LTO 分数画在 1393 年有效的 CHGIS 多边形上。
# st_make_valid 整理多边形几何，供 geom_sf 绘制。
lto <- read.csv(file.path(root, "data/LTO_fu_scores.csv"))
hist_pref <- st_read(file.path(root, "data/chgis地图多边形/v6_time_pref_pgn_utf_wgs84/v6_time_pref_pgn_utf_wgs84.shp"), quiet = TRUE) %>%
  filter(BEG_YR <= 1393, END_YR >= 1393) %>%
  st_transform(4326) %>%
  st_make_valid()
# 按 SYS_ID 连接分数，保留地图中的所有多边形；没有分数的多边形使用默认缺失值颜色。
# 白色到黑色表示 LTO 从低到高；政区边界使用灰色细线，图例标为 LTO。
map_data <- left_join(hist_pref, select(lto, SYS_ID, LTO), by = "SYS_ID", relationship = "one-to-one")

# 地图采用 Albers 等积投影、Arial 11 号字体和 classic 主题。
# 由全部政区的投影边界确定显示范围，四周对称留 2% 空白，使地图主体居中。
# 隐去坐标轴、网格和外框；横向色标放在右下方空白处，标题在色标上方。
font_add("Arial", "Arial.ttf")
projection <- "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +ellps=GRS80 +units=m +no_defs"
extent <- st_bbox(st_transform(map_data, projection))
padding <- .02 * max(extent["xmax"] - extent["xmin"], extent["ymax"] - extent["ymin"])
p_map <- ggplot(map_data) +
  geom_sf(aes(fill = LTO), color = "grey60", linewidth = .1) +
  scale_fill_gradient(high = "black", low = "white", name = "LTO") +
  guides(fill = guide_colorbar(direction = "horizontal", title.position = "top")) +
  coord_sf(crs = projection, datum = NA, expand = FALSE,
           xlim = extent[c("xmin", "xmax")] + c(-1, 1) * padding,
           ylim = extent[c("ymin", "ymax")] + c(-1, 1) * padding) +
  theme_classic(base_family = "Arial", base_size = 11) +
  theme(axis.title = element_blank(), axis.text = element_blank(),
        axis.ticks = element_blank(), axis.line = element_blank(),
        legend.position = "inside", legend.position.inside = c(.97, .05),
        legend.justification = c(1, 0), legend.box = "vertical",
        legend.background = element_rect(fill = "white", color = NA))
ggsave(file.path(root, "output/lto_map.pdf"), p_map, width = 7, height = 6)
