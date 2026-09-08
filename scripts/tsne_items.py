"""根据已保存的题项向量重新生成两份 t-SNE 数据表。"""

import os
from pathlib import Path

# 限制 joblib 的线程探测范围；t-SNE 通过固定随机种子保持可复现。
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parents[1]
SENTENCES = ROOT / "data/LTO_sentences.csv"
EMBEDDINGS = ROOT / "data/LTOitem_embeddings"


def save_tsne(data, vectors, output):
    """降维向量，并把坐标与题项标签一起写入 CSV。"""
    coordinates = TSNE(n_components=2, perplexity=10, random_state=42, init="pca", n_jobs=1).fit_transform(vectors)
    # 将模块代码翻译成 R 图例使用的中文题项文本；输出行仍保持向量顺序。
    items = pd.read_csv(SENTENCES).loc[lambda x: ~x.is_guwen].set_index("模块")["句子"]
    pd.DataFrame({
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
        "label": data["模块"].to_numpy(),
        "is_guwen": data["is_guwen"].to_numpy(),
        "text": data["句子"].to_numpy(),
        "items": data["模块"].map(items).to_numpy(),
    }).to_csv(EMBEDDINGS / output, index=False, encoding="utf-8")


# Qwen 向量覆盖全部 44 个题项；古文向量覆盖第二张图使用的 40 条古文句子。
sentences = pd.read_csv(SENTENCES)
save_tsne(
    sentences,
    np.load(EMBEDDINGS / "lto_qwen3_embeddings.npy"),
    "lto_qwen3_tsne.csv",
)
ancient = sentences.loc[sentences.is_guwen].copy()
save_tsne(
    ancient,
    np.load(EMBEDDINGS / "lto_ancient_embeddings.npy"),
    "lto_ancient_tsne.csv",
)
