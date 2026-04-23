"""Модуль для работы с эмбеддингами через локальную Ollama."""
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional
import time

try:
    import ollama
except ImportError:
    raise ImportError("Установите ollama: pip install ollama")


class LocalEmbedder:
    """
    Класс для создания эмбеддингов текста через локальные модели Ollama.
    Поддерживает кэширование для ускорения повторных запросов.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        cache_dir: str = "./data/embeddings_cache",
        use_cache: bool = True
    ):
        """
        Инициализация эмбеддера.

        Args:
            model_name: название модели в Ollama (nomic-embed-text, mxbai-embed-large, all-minilm)
            cache_dir: директория для кэша эмбеддингов
            use_cache: использовать ли кэширование
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Проверяем доступность модели
        self._check_model()

        # Счётчики для статистики
        self.cache_hits = 0
        self.cache_misses = 0

    def _check_model(self) -> None:
        """Проверяет, доступна ли модель в Ollama."""
        try:
            response = ollama.list()
            available_models = []
            
            if hasattr(response, 'models'):
                for m in response.models:
                    available_models.append(m.model)
            elif isinstance(response, dict) and 'models' in response:
                for m in response['models']:
                    available_models.append(m.get('name') or m.get('model'))
            elif isinstance(response, list):
                for m in response:
                    available_models.append(m.get('name') or m.get('model'))
            
            model_exists = False
            base_name = self.model_name.split(':')[0]
            for m in available_models:
                if self.model_name == m or m.startswith(f"{self.model_name}:") or base_name == m.split(':')[0]:
                    model_exists = True
                    self.model_name = m
                    break
            
            if not model_exists:
                print(f"⚠️  Модель '{self.model_name}' не найдена локально.")
                print(f"   Выполните: ollama pull {self.model_name}")
                print()
        except Exception as e:
            print(f"⚠️  Не удалось проверить модель: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Создаёт ключ кэша на основе текста и модели."""
        content = f"{self.model_name}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Возвращает путь к файлу кэша."""
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Загружает эмбеддинг из кэша, если он существует."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache_hits += 1
                    return data['embedding']
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def _save_to_cache(self, cache_key: str, embedding: List[float]) -> None:
        """Сохраняет эмбеддинг в кэш."""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'model': self.model_name,
                    'embedding': embedding,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            print(f"⚠️  Ошибка сохранения в кэш: {e}")

    def embed(self, text: str, retries: int = 3) -> List[float]:
        """
        Создаёт эмбеддинг для текста.

        Args:
            text: текст для векторизации
            retries: количество повторных попыток при ошибке

        Returns:
            список чисел (вектор)
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # Проверяем кэш
        cache_key = self._get_cache_key(text)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        self.cache_misses += 1

        # Запрос к Ollama
        for attempt in range(retries):
            try:
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                embedding = response['embedding']

                # Сохраняем в кэш
                self._save_to_cache(cache_key, embedding)

                return embedding

            except Exception as e:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Не удалось создать эмбеддинг после {retries} попыток: {e}"
                    )
                print(f"⚠️  Попытка {attempt + 1} не удалась: {e}. Повтор...")
                time.sleep(1)

        return []  # Никогда не выполнится, но для типизации

    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Создаёт эмбеддинги для списка текстов.

        Args:
            texts: список текстов
            show_progress: показывать прогресс

        Returns:
            список векторов
        """
        embeddings = []
        total = len(texts)

        for i, text in enumerate(texts):
            if show_progress and total > 1:
                progress = (i + 1) / total * 100
                print(f"\r🔄 Создание эмбеддингов: {progress:.1f}%", end="", flush=True)

            embedding = self.embed(text)
            embeddings.append(embedding)

        if show_progress and total > 1:
            print("\r✅ Эмбеддинги созданы" + " " * 20)

        return embeddings

    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику использования кэша."""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': self.cache_hits + self.cache_misses
        }

    def clear_cache(self) -> None:
        """Очищает кэш эмбеддингов."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        print(f"🧹 Кэш эмбеддингов очищен ({self.cache_dir})")


# Пример использования
if __name__ == "__main__":
    # Тестирование эмбеддера
    print("🔍 Тестирование LocalEmbedder")
    print("=" * 50)

    embedder = LocalEmbedder(model_name="nomic-embed-text")

    # Тест 1: Одиночный текст
    text1 = "Python - отличный язык программирования"
    print(f"\n📝 Текст: '{text1}'")
    embedding = embedder.embed(text1)
    print(f"✅ Размер вектора: {len(embedding)}")
    print(f"📊 Первые 5 значений: {embedding[:5]}")

    # Тест 2: Кэширование
    print("\n🔄 Проверка кэширования...")
    start = time.time()
    embedding2 = embedder.embed(text1)
    time1 = time.time() - start

    start = time.time()
    embedding3 = embedder.embed(text1)
    time2 = time.time() - start

    print(f"⏱️  Первый запрос: {time1:.3f}с")
    print(f"⏱️  Второй запрос (из кэша): {time2:.3f}с")
    print(f"📊 Статистика: {embedder.get_stats()}")

    # Тест 3: Batch
    texts = [
        "Искусственный интеллект",
        "Машинное обучение",
        "Нейронные сети"
    ]
    print(f"\n📚 Batch-обработка {len(texts)} текстов...")
    batch_embeddings = embedder.embed_batch(texts)
    print(f"✅ Создано {len(batch_embeddings)} векторов")