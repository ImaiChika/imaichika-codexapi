from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    import chromadb
except Exception:  
    chromadb = None


class VectorStore:
    """
    Chroma-backed vector store with in-memory fallback.
    """

    def __init__(self, persist_dir: Path, collection_name: str = "chat_messages"):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._use_chroma = False
        self._collection = None
        self._fallback: List[Dict] = []

        if chromadb is None:
            return

        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    @staticmethod
    def _to_list(vec) -> Optional[List[float]]:
        if vec is None:
            return None
        if isinstance(vec, torch.Tensor):
            return vec.detach().cpu().tolist()
        if isinstance(vec, (list, tuple)):
            return list(vec)
        return None

    def add(self, doc_id: str, text: str, embedding, metadata: Optional[Dict] = None):
        print(f"Adding to Chroma: {self._use_chroma}")
        if not text:
            return
        vec_list = self._to_list(embedding)
        if vec_list is None:
            return

        if self._use_chroma and self._collection is not None:
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[vec_list],
                documents=[text],
                metadatas=[metadata or {}],
            )
            return

        self._fallback.append(
            {
                "id": doc_id,
                "text": text,
                "embedding": embedding if isinstance(embedding, torch.Tensor) else torch.tensor(vec_list),
                "metadata": metadata or {},
            }
        )

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = torch.norm(a) * torch.norm(b)
        if denom.item() == 0:
            return 0.0
        return float(torch.dot(a, b) / denom)

    def query(self, embedding, top_k: int = 8) -> List[Dict]:
        if embedding is None:
            return []

        vec_list = self._to_list(embedding)
        if vec_list is None:
            return []

        if self._use_chroma and self._collection is not None:
            res = self._collection.query(
                query_embeddings=[vec_list],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            ids = res.get("ids", [[]])[0] if "ids" in res else [""] * len(docs)

            out = []
            for doc, meta, dist, rid in zip(docs, metas, dists, ids):
                sim = 1.0 - float(dist) if dist is not None else 0.0
                out.append(
                    {
                        "id": rid,
                        "text": doc or "",
                        "metadata": meta or {},
                        "score": sim,
                    }
                )
            return out

        # fallback in-memory search
        emb = embedding if isinstance(embedding, torch.Tensor) else torch.tensor(vec_list)
        scored = []
        for item in self._fallback:
            sim = self._cosine_similarity(emb, item["embedding"])
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for sim, item in scored[:top_k]:
            out.append(
                {
                    "id": item.get("id", ""),
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                    "score": float(sim),
                }
            )
        return out
