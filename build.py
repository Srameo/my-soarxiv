import json, numpy as np
from sentence_transformers import SentenceTransformer
import umap

# 1. 读数据
papers = json.load(open("papers.json"))
texts  = [p["title"] + " " + p["abstract"] for p in papers]

# 2. 768 维语义向量
model = SentenceTransformer("all-MiniLM-L6-v2")          # 15 篇速度毫秒级
emb   = model.encode(texts, normalize_embeddings=True)

# 3. 降到 3D
coords = umap.UMAP(n_components=3, random_state=42).fit_transform(emb)

# 4. 写给前端
out = [
    {**p, "pos": coords[i].round(4).tolist()} for i, p in enumerate(papers)
]
json.dump(out, open("galaxy.json", "w"), ensure_ascii=False, indent=2)
print("✨ 生成 galaxy.json OK，共", len(out), "篇")