"""RAG модуль для локального поиска по документам."""
from .embedder import LocalEmbedder
from .vector_store import VectorStore
from .document_loader import DocumentLoader

__all__ = ['LocalEmbedder', 'VectorStore', 'DocumentLoader']