# save as test_search.py
import sys
sys.path.append('.')
from rag import LocalEmbedder, VectorStore

embedder = LocalEmbedder("nomic-embed-text")
store = VectorStore()
store.load("index_test_doc")

query = "Что такое Ollama?"
print(f"Поиск: '{query}'")

query_vector = embedder.embed(query)
results = store.search(query_vector, top_k=3)

print(f"\nНайдено результатов: {len(results)}")
for i, (text, meta, score) in enumerate(results, 1):
    print(f"{i}. [score: {score:.3f}] {text[:100]}...")