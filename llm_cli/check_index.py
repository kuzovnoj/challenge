# Создадим скрипт для проверки индекса
# save as check_index.py
import sys
sys.path.append('.')
from rag.vector_store import VectorStore

store = VectorStore()
store.load("index_test_doc")

print(f"Загружено чанков: {len(store.chunks)}")
for i, chunk in enumerate(store.chunks):
    print(f"\nЧанк {i+1}:")
    print(f"  Текст: {chunk}")
    print(f"  Метаданные: {store.metadata[i]}")