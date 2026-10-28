"""计算各政区的长期取向分数。"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
# 输入包括地方条目、CHGIS 历史多边形及两者的对应表。
DOCUSKY = ROOT / "data/明一统志文本/明一统志文本_docusky"
CHGIS = ROOT / "data/chgis地图多边形/v6_time_pref_pgn_utf_wgs84/v6_time_pref_pgn_utf_wgs84.shp"
SOURCES = ROOT / "data/明一统志文本/chgis-docusky文件映射表.csv"
MODEL = "Jihuai/bert-ancient-chinese"
# revision 固定模型权重和分词器的版本。
REVISION = "1eb4c2023b541119a81c39be9151ca292f836492"
YEAR = 1393
# BERT 共容纳 512 个 token，正文占 510 个，另外两个位置留给首尾特殊标记。
# 相邻窗口前进 128 个 token，重叠部分保留局部上下文。
LENGTH = 510
STRIDE = 128
# 四个模块对应 VSM 的第 13、14、19、22 题，每组含 10 条古文比较句。
MODULES = {"m13": "交友", "m14": "节俭", "m19": "国家自豪", "m22": "坚持"}


def split(text, tokenizer):
    """窗口长 510 个 token，步长 128。"""
    # 先对全文分词，再按 token 位置切片；切窗时不计入模型添加的特殊标记。
    # 最后一个窗口可以短于 510 个 token；短文本直接构成一个窗口。
    # decode 将各段 token 转回字符串，交给 SentenceTransformer 批量编码。
    ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode(ids[start:start + LENGTH])
            for start in range(0, max(1, len(ids) - LENGTH + STRIDE), STRIDE)]


def main():
    # 确定分析政区。保留起止年份包含 1393 年的多边形，排除“非政区国土”。
    # SYS_ID 是连接文本和地图的政区标识，NAME_CH 是分数表中的政区名称。
    # 映射表按这批 SYS_ID 取行，再与多边形一一对应。
    polygons = gpd.read_file(CHGIS)
    polygons = polygons.loc[(polygons.BEG_YR <= YEAR) & (polygons.END_YR >= YEAR)
                            & polygons.TYPE_CH.ne("非政区国土"), ["SYS_ID", "NAME_CH"]]
    mapping = pd.read_csv(SOURCES).set_index("SYS_ID")
    mapping = mapping.loc[polygons.SYS_ID].reset_index()
    scores = polygons.merge(mapping, on="SYS_ID", how="left", validate="one_to_one")

    # 编码比较句。四组古文用于计分，另四条题项原文用于题项图。
    # 随机种子为 42；计算设备依次选择 Apple GPU、CUDA GPU、CPU。
    # normalize_embeddings 将每个向量归一化为单位长度，向量点积即余弦相似度。
    torch.manual_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, revision=REVISION, device=device)
    items = pd.read_csv(ROOT / "data/LTO_sentences.csv")
    item_vectors = model.encode(items.句子.tolist(), normalize_embeddings=True)
    # 向量、句子、模块和古文标记按相同行序保存，供后面的 t-SNE 绘图读取。
    np.savez(ROOT / "data/LTOitem_embeddings/lto_items.npz", vectors=item_vectors,
             text=items.句子.to_numpy(dtype=str), module=items.模块.to_numpy(dtype=str),
             is_guwen=items.is_guwen.to_numpy())
    ancient = items.loc[items.is_guwen]
    ancient_vectors = item_vectors[items.is_guwen]

    # 计算地方条目与四组古文的相似度。同一来源只编码一次，供对应政区共用。
    # vectors 的每行是一个正文窗口，ancient_vectors 的每行是一条古文比较句。
    # 两者相乘得到“窗口数 × 40”的相似度矩阵；每组对全部窗口和 10 条古文取平均。
    source_means = []
    for source_path, units in scores.groupby("source_path", sort=False):
        text = (DOCUSKY / source_path).read_text(encoding="utf-8").strip()
        windows = split(text, model.tokenizer)
        vectors = model.encode(windows, batch_size=16, normalize_embeddings=True)
        # 每个政区各保存一份窗口向量，文件位置沿用映射表。
        for path in units.embedding_path:
            embedding = ROOT / path
            embedding.parent.mkdir(parents=True, exist_ok=True)
            np.save(embedding, vectors)
        similarity = vectors @ ancient_vectors.T
        means = [similarity[:, ancient.模块.eq(module)].mean() for module in MODULES]
        source_means.append([source_path, len(windows), *means])
        # 释放该条目占用的 Apple GPU 缓存，再处理下一条。
        if device == "mps":
            torch.mps.empty_cache()

    # 将条目均值对应到各政区，再对每个模块计算 z 分数。
    # 每个政区在标准化中占一行、权重相同；使用总体标准差（ddof=0）。
    # m13_交友等四列保存标准化后的模块分数。
    columns = [f"{code}_{name}" for code, name in MODULES.items()]
    source_means = pd.DataFrame(source_means, columns=["source_path", "windows", *columns])
    scores = scores.merge(source_means, on="source_path", how="left", validate="many_to_one")
    similarity = scores[columns]
    scores[columns] = (similarity - similarity.mean()) / similarity.std(ddof=0)

    # 按 VSM 的方向和权重合成长期取向分数。
    # 节俭减交友的差值乘 40，坚持减国家自豪的差值乘 25，两项相加为 LTO_raw。
    # 所有政区统一减去最小值，得到从 0 开始的 LTO；平移保持政区之间的分差和排序。
    # 分数表包含政区标识、名称、四个模块分数、LTO_raw、窗口数和 LTO。
    scores["LTO_raw"] = 40 * (scores.m14_节俭 - scores.m13_交友) + 25 * (scores.m22_坚持 - scores.m19_国家自豪)
    scores["LTO"] = scores.LTO_raw - scores.LTO_raw.min()
    scores = scores[["SYS_ID", "NAME_CH", *columns, "LTO_raw", "windows", "LTO"]]
    scores.to_csv(ROOT / "data/LTO_fu_scores.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
