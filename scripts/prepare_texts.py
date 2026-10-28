"""从 DocuSky XML 提取地方条目，再按映射表生成各政区的文本。"""

from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
# 两类文本分别保存：DocuSky 文件使用原条目名，CHGIS 文件使用地图中的政区名。
TEXTS = ROOT / "data/明一统志文本"
DOCUSKY = TEXTS / "明一统志文本_docusky"


def main():
    # XML 分为两份，documents 下的每个 document 是一条文献。
    # doc_source 为“大明”的文献是国内条目；docclass、doctype 分别提供目录名和条目名。
    # doc_content 下的子元素对应原段落，itertext() 同时提取正文及嵌套的夹注。
    # 段落内部的连续空白合并为空格，各段落之间换行；文字、标点、缺字标记照录。
    for xml in sorted(DOCUSKY.glob("*.xml")):
        for document in ET.parse(xml).getroot().find("documents"):
            if document.findtext("doc_source") != "大明":
                continue
            province = document.findtext("docclass")
            name = document.findtext("doctype")
            paragraphs = document.find("doc_content")
            text = "\n".join(" ".join("".join(p.itertext()).split()) for p in paragraphs)
            path = DOCUSKY / province / f"{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n", encoding="utf-8")

    # 映射表每行对应一个 CHGIS 政区，source_path 指向它使用的 DocuSky 条目。
    # embedding_path 的目录和文件名沿用政区名称，将 .npy 换成 .txt 即为政区文本路径。
    # 多个政区使用同一条目时，分别保存该条目的完整正文。
    mapping = pd.read_csv(TEXTS / "chgis-docusky文件映射表.csv")
    for row in mapping.itertuples():
        text = (DOCUSKY / row.source_path).read_text(encoding="utf-8").strip()
        relative = Path(row.embedding_path).relative_to("data/fu_embeddings").with_suffix(".txt")
        path = TEXTS / "明一统志文本_chgis" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
