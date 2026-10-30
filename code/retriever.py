"""Dense retriever (bge-small-en-v1.5 + FAISS inner-product index over normalised vectors).

Building the index: .venv\\Scripts\\python.exe code\\retriever.py hotpotqa   (cached to data/<name>/index.faiss)
"""
import sys
import time

import faiss
import numpy as np

from common import ROOT, load_config, read_jsonl, write_json


class Retriever:
    def __init__(self, dataset, cfg, device="cuda"):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg["retriever"]
        self.dir = ROOT / "data" / dataset
        self.passages = read_jsonl(self.dir / "corpus.jsonl")
        self.model = SentenceTransformer(self.cfg["model"], device=device)
        index_path = self.dir / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            self.index = self._build(index_path)

    def _build(self, index_path):
        t0 = time.perf_counter()
        vecs = self.model.encode([p["text"] for p in self.passages], batch_size=128,
                                 normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs.astype(np.float32))
        faiss.write_index(index, str(index_path))
        write_json(self.dir / "index_stats.json", {"passages": len(self.passages), "dim": int(vecs.shape[1]),
                                                   "build_seconds": round(time.perf_counter() - t0, 1)})
        return index

    def search(self, query, k):
        q = self.model.encode([self.cfg["query_instruction"] + query], normalize_embeddings=True,
                              show_progress_bar=False)
        scores, idx = self.index.search(q.astype(np.float32), k)
        return [{**self.passages[i], "score": float(s)} for s, i in zip(scores[0], idx[0])]

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


if __name__ == "__main__":
    r = Retriever(sys.argv[1], load_config())
    print(len(r.passages), "passages indexed")
