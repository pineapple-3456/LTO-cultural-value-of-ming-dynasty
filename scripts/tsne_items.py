"""生成 BERT 古文图和 Qwen 题项图的 t-SNE 数据。"""

import os
from pathlib import Path

# t-SNE 使用一个工作进程，joblib 的 CPU 探测范围也设为一个。
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "data/LTOitem_embeddings"
MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def save_tsne(data, vectors, output):
    # 将每条句子的模型向量降为两个坐标，用于观察题项与古文的相对位置。
    # 余弦距离与计分时的相似度对应，比较句子的语义方向。
    # perplexity=10 控制邻域规模；坐标随机初始化，随机种子固定为 42。
    coordinates = TSNE(n_components=2, metric="cosine", perplexity=10, random_state=42,
                       init="random", n_jobs=1).fit_transform(vectors)
    # 坐标与输入句子逐行对应，连同模块、古文标记和题项文字写入绘图数据。
    data = data.copy()
    data[["x", "y"]] = coordinates
    data.to_csv(EMBEDDINGS / output, index=False, encoding="utf-8-sig")


def main():
    # 读取计分时保存的同一组向量及其标签。
    # is_guwen=True 是 40 条古文，False 是四条题项原文。
    archive = np.load(EMBEDDINGS / "lto_items.npz")
    data = pd.DataFrame({"label": archive["module"], "text": archive["text"],
                         "is_guwen": archive["is_guwen"]})
    # 每个模块的题项原文作为中文图例，通过模块代码对应到所有古文句子。
    labels = data.loc[~data.is_guwen].set_index("label").text
    data["items"] = data.label.map(labels)
    # BERT 图使用计分时保存的 40 条古文向量，观察四组比较句的分布。
    save_tsne(data.loc[data.is_guwen], archive["vectors"][data.is_guwen], "lto_ancient_tsne.csv")

    # Qwen 将相同的 40 条古文和四条题项原文共同编码，观察原题与对应古文的位置。
    # 直接编码句子，不添加检索指令；模型自带末位 token 池化，输出归一化的 1024 维向量。
    # float32 和 eager 注意力支持本机 Apple GPU 上的分组查询注意力计算。
    torch.manual_seed(42)
    model = SentenceTransformer(MODEL, revision=REVISION,
                               model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"})
    vectors = model.encode(data.text.tolist(), batch_size=8, normalize_embeddings=True)
    # 与 BERT 文件一样，向量和句子标签按相同行序保存。
    np.savez(EMBEDDINGS / "lto_qwen3_items.npz", vectors=vectors, text=archive["text"],
             module=archive["module"], is_guwen=archive["is_guwen"])
    save_tsne(data, vectors, "lto_all_tsne.csv")


if __name__ == "__main__":
    main()
