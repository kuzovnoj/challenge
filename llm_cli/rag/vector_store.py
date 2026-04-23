"""Векторное хранилище на основе NumPy с косинусным сходством."""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class VectorStore:
    def __init__(self, persist_dir: str = "./rag_data/vectors"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vectors: Optional[np.ndarray] = None
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.normalized_vectors: Optional[np.ndarray] = None

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def add(self, chunks: List[str], vectors: List[List[float]],
            metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Количество чанков и векторов должно совпадать")
        if metadata is None:
            metadata = [{} for _ in chunks]
        new_vectors = np.array(vectors, dtype=np.float32)
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        self.normalized_vectors = self._normalize(self.vectors)

    def search(self, query_vector: List[float], top_k: int = 5,
               min_similarity: float = 0.0) -> List[Tuple[str, Dict[str, Any], float]]:
        if self.vectors is None or len(self.vectors) == 0:
            return []
        query = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        query_norm = self._normalize(query)
        similarities = np.dot(query_norm, self.normalized_vectors.T).flatten()
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= min_similarity:
                results.append((self.chunks[idx], self.metadata[idx].copy(), sim))
        return results[:top_k]

    def get_sources(self) -> List[str]:
        sources = set()
        for meta in self.metadata:
            if 'source' in meta:
                sources.add(meta['source'])
        return sorted(list(sources))

    def delete_by_source(self, source: str) -> int:
        if self.vectors is None:
            return 0
        keep_indices = [i for i, meta in enumerate(self.metadata)
                        if meta.get('source') != source]
        deleted = len(self.metadata) - len(keep_indices)
        if deleted > 0:
            self.chunks = [self.chunks[i] for i in keep_indices]
            self.metadata = [self.metadata[i] for i in keep_indices]
            self.vectors = self.vectors[keep_indices]
            self.normalized_vectors = self._normalize(self.vectors)
        return deleted

    def save(self, name: str = "main_index") -> None:
        if self.vectors is None:
            print("⚠️  Нет данных для сохранения")
            return
        vectors_path = self.persist_dir / f"{name}_vectors.npy"
        np.save(vectors_path, self.vectors)
        data_path = self.persist_dir / f"{name}_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({'chunks': self.chunks, 'metadata': self.metadata}, f)
        meta_path = self.persist_dir / f"{name}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'vector_count': len(self.vectors),
                'vector_dim': self.vectors.shape[1],
                'total_chunks': len(self.chunks)
            }, f, indent=2)
        print(f"✅ Индекс сохранён в {self.persist_dir} ({len(self.vectors)} векторов)")

    def load(self, name: str = "main_index") -> bool:
        vectors_path = self.persist_dir / f"{name}_vectors.npy"
        data_path = self.persist_dir / f"{name}_data.pkl"
        if not vectors_path.exists() or not data_path.exists():
            return False
        try:
            self.vectors = np.load(vectors_path)
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            self.normalized_vectors = self._normalize(self.vectors)
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки индекса: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        if self.vectors is None:
            return {'total_vectors': 0, 'total_chunks': 0,
                    'vector_dimension': 0, 'sources': [], 'memory_usage_mb': 0}
        sources = list(set(meta.get('source', 'unknown') for meta in self.metadata))
        return {
            'total_vectors': len(self.vectors),
            'total_chunks': len(self.chunks),
            'vector_dimension': self.vectors.shape[1],
            'sources': sources,
            'memory_usage_mb': self.vectors.nbytes / (1024 * 1024)
        }

    def clear(self) -> None:
        self.vectors = None
        self.normalized_vectors = None
        self.chunks = []
        self.metadata = []
        print("🧹 Векторное хранилище очищено")