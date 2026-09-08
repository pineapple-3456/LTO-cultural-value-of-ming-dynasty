"""计算 1393 年历史政区的长期取向分数。

脚本优先复用已有的文本窗口向量；只有缓存缺失或设置
``REBUILD_EMBEDDINGS=1`` 时才加载本地模型，最后写出供 ``scripts/plot.R``
使用的分数表。
"""

import os
from pathlib import Path

# 模型应已存在于本地 Hugging Face 缓存中；强制离线可避免运行时意外下载。
os.environ["HF_HUB_OFFLINE"] = "1"

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# 所有输入都从项目根目录解析，因此可从任意当前目录启动脚本。
ROOT = Path(__file__).resolve().parents[1]
DOCUSKY = ROOT / "data/明一统志文本/明一统志文本_docusky"
CHGIS = ROOT / "data/chgis地图多边形/v6_time_pref_pgn_utf_wgs84/v6_time_pref_pgn_utf_wgs84.shp"
SOURCES = ROOT / "data/明一统志文本/chgis-docusky文件映射表.csv"
SENTENCES = ROOT / "data/LTO_sentences.csv"
ITEM_EMBEDDINGS = ROOT / "data/LTOitem_embeddings/lto_ancient_embeddings.npy"
SCORES = ROOT / "data/LTO_fu_scores.csv"

MODEL = "Jihuai/bert-ancient-chinese"
YEAR = 1393
# BERT 要为特殊 token 保留两个位置，因此正文窗口设为 510，正好不超过
# 模型的 512 位长度上限。
LENGTH = 510
STRIDE = 128
# 这四个模块用于后面的 VSM 风格长期取向组合公式。
MODULES = ("m13", "m14", "m19", "m22")
# 全量重算很慢，只有显式设置环境变量时才重建全部缓存。
REBUILD = os.getenv("REBUILD_EMBEDDINGS") == "1"


def split(text, tokenizer):
    """把一份原文切成有重叠、且适合模型长度的文本窗口。"""
    ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    return [
        tokenizer.decode(ids[i : i + LENGTH], skip_special_tokens=True)
        for i in range(0, max(1, len(ids) - LENGTH + STRIDE), STRIDE)
    ]


def encode(model, texts):
    """批量编码并归一化向量，便于后面直接计算余弦相似度。"""
    vectors = model.encode(texts, batch_size=32, convert_to_numpy=True)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def main():
    # 只保留 1393 年有效的政区，并排除非行政区国土多边形。东番会在 R
    # 地图中保留，但不进入分数计算。
    polygons = gpd.read_file(CHGIS)
    polygons = polygons[(polygons.BEG_YR <= YEAR) & (polygons.END_YR >= YEAR)]
    polygons = polygons[polygons.TYPE_CH != "非政区国土"]
    polygons["SYS_ID"] = polygons["SYS_ID"].astype(int)
    # 映射表是政区与原文之间唯一的对应关系；历史文本相同时，多个政区
    # 可以共用同一个 source_path。
    scores = polygons[["SYS_ID", "NAME_CH"]].merge(pd.read_csv(SOURCES), on="SYS_ID")
    scores["embedding_path"] = scores.embedding_path.map(lambda path: ROOT / path)

    # 文言比较句与已保存向量按行对应；只有 40 条古文句子进入四个模块
    # 的平均相似度计算。
    items = pd.read_csv(SENTENCES)
    items = items[items.is_guwen]
    item_vectors = np.load(ITEM_EMBEDDINGS)
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)
    # 默认只补缺失缓存；显式要求时才全量重建。所有缺失窗口一次批量编码，
    # 避免逐文件重复加载模型。
    missing = scores if REBUILD else scores[~scores.embedding_path.map(lambda path: Path(path).exists())]
    if len(missing):
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        model = SentenceTransformer(
            MODEL,
            device=device,
            local_files_only=True,
            model_kwargs={"attn_implementation": "eager", "torch_dtype": "float32"},
        )
        windows = []
        batches = []
        for row in missing.itertuples():
            # 一个映射行可能包含多个文本文件，按映射表顺序拼接后再切窗。
            text = "\n".join(
                (DOCUSKY / path).read_text().strip()
                for path in row.source_path.split(";")
            )
            chunks = split(text, tokenizer)
            batches.append((Path(row.embedding_path), len(windows), len(chunks)))
            windows.extend(chunks)
        vectors = encode(model, windows)
        for embedding_path, start, length in batches:
            embedding_path.parent.mkdir(parents=True, exist_ok=True)
            # 先写临时文件再原子替换，避免中断后留下看似有效的半个缓存。
            temporary = embedding_path.with_name(embedding_path.name + ".tmp.npy")
            np.save(temporary, vectors[start : start + length])
            os.replace(temporary, embedding_path)

    # 每个不同的 source embedding 只计算一次；共用原文的政区因此得到同一
    # 个底层文本测量值。
    source_scores = {}
    for embedding in scores.embedding_path.unique():
        vectors = np.load(embedding)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        similarity = vectors @ item_vectors.T
        source_scores[embedding] = np.column_stack(
            [similarity[:, items.模块.eq(module)].mean(axis=1) for module in MODULES]
        )

    unit_scores = [source_scores[embedding] for embedding in scores.embedding_path]
    all_scores = np.vstack(unit_scores)
    # 在全部映射政区上标准化四个模块，再按 VSM 风格计算：第一组差值权重
    # 为 40，第二组差值权重为 25。
    mapped = 3 - 0.5 * (all_scores - all_scores.mean(axis=0)) / all_scores.std(axis=0)
    offset = 0
    values = []
    for unit in unit_scores:
        mean = mapped[offset : offset + len(unit)].mean(axis=0)
        offset += len(unit)
        values.append((*mean, 40 * (mean[0] - mean[1]) + 25 * (mean[2] - mean[3]), len(unit)))

    scores[["m13_交友", "m14_节俭", "m19_国家自豪", "m22_坚持", "LTO_raw", "windows"]] = values
    # 平移原始指数，使最终报告的 LTO 从零开始。
    scores["LTO"] = scores.LTO_raw - scores.LTO_raw.min()
    scores = scores.drop(columns=["source_path", "embedding_path"])
    scores.to_csv(SCORES, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
