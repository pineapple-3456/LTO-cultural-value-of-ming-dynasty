# 明代长期取向地图

使用明一统志文本计算明代各府 (直隶州、卫等) 的长期取向分数。文本来自[DocuSky](https://docusky.org.tw/DocuSky/projects/ntu/DaMingYiTongZhi/index.html)的四库全书版《明一统志》。地图来自[CHGIS](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/I0Q7SM) 中 1393 年（洪武26年）府多边形。

原始文本存储在 `data/明一统志文本/明一统志文本_docusky`，经整理后的文本在 `data/明一统志文本/明一统志文本_chgis`，其中每个文件对应CHGIS中的一个府，缺失的府用上级条目代替，例如甘州卫、山丹卫等文件中实际的内容是对应陕西行都指挥使司，具体参见 `data/明一统志文本/chgis-docusky文件映射表.csv`。

使用 bert-ancient-chinese 对各府的文本进行滑动窗口嵌入，计算与 40 个长期取向相关文言文句子的相似度，根据 [VALUES SURVEY MODULE 2013](https://geerthofstede.com/research-and-vsm/vsm-2013/) 的计算方式合成长期取向分数。结果存储在 `data/LTO_fu_scores.csv`。