#!/usr/bin/env python3
"""
Простое CLI-приложение для общения с локальной LLM через Ollama.
С поддержкой RAG (Retrieval-Augmented Generation) с постоянным файловым хранилищем.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    import ollama
except ImportError:
    print("Ошибка: библиотека ollama не установлена.")
    print("Установите её командой: pip install ollama")
    sys.exit(1)

# Импортируем RAG модули
try:
    from rag import LocalEmbedder, VectorStore, DocumentLoader
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  RAG модули не найдены: {e}")
    print("   RAG функциональность будет отключена")
    RAG_AVAILABLE = False

RAG_DEFAULT_DIR = Path("./rag_data")
RAG_INDEX_NAME = "main_index"


class LocalLLMChat:
    """Класс для управления чатом с локальной LLM и RAG."""

    def __init__(
        self,
        model: str,
        no_rag: bool = False,
        embed_model: str = "nomic-embed-text",
        top_k: int = 3,
        min_similarity: float = 0.3
    ):
        """
        Инициализация чата.

        Args:
            model: название модели в Ollama
            no_rag: если True, отключает RAG даже при наличии индекса
            embed_model: модель для эмбеддингов
            top_k: количество релевантных чанков для контекста
            min_similarity: минимальное сходство для включения в контекст
        """
        self.model = model
        self.messages: List[Dict[str, str]] = []
        self.rag_enabled = False
        self.embedder = None
        self.vector_store = None
        self.top_k = top_k
        self.min_similarity = min_similarity

        if not no_rag and RAG_AVAILABLE:
            # Пытаемся загрузить существующий индекс из папки по умолчанию
            self._init_rag(str(RAG_DEFAULT_DIR), embed_model)

        self._check_model_availability()

    def _check_model_availability(self) -> None:
        """Проверяет, доступна ли указанная модель в Ollama."""
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
            base_name = self.model.split(':')[0]
            for m in available_models:
                if self.model == m or m.startswith(f"{self.model}:") or base_name == m.split(':')[0]:
                    model_exists = True
                    self.model = m
                    break

            if not model_exists:
                print(f"❌ Модель '{self.model}' не найдена локально!")
                print(f"   Установите её командой: ollama pull {self.model}")
                if available_models:
                    print(f"   Доступные модели: {', '.join(available_models)}")
                else:
                    print("   Доступные модели: нет")
                sys.exit(1)
            else:
                print(f"✅ Модель '{self.model}' найдена")
        except Exception as e:
            print(f"⚠️  Не удалось проверить список моделей: {e}")

    def _init_rag(self, rag_path: str, embed_model: str) -> None:
        """Инициализирует RAG с хранилищем по указанному пути."""
        rag_path = Path(rag_path)
        rag_path.mkdir(parents=True, exist_ok=True)

        self.embedder = LocalEmbedder(model_name=embed_model)
        self.vector_store = VectorStore(persist_dir=str(rag_path / "vectors"))

        if self.vector_store.load(RAG_INDEX_NAME):
            stats = self.vector_store.get_stats()
            print(f"✅ RAG активирован. Загружен индекс: {stats['total_vectors']} векторов, {len(stats['sources'])} источников.")
        else:
            print("ℹ️ RAG готов к работе. Индекс пока пуст. Используйте /rag add для добавления документов.")

        self.rag_enabled = True

    def _get_rag_context(self, query: str) -> Optional[str]:
        """Получает релевантный контекст из RAG."""
        if not self.rag_enabled or not self.vector_store:
            return None

        query_vector = self.embedder.embed(query)
        results = self.vector_store.search(
            query_vector,
            top_k=self.top_k,
            min_similarity=self.min_similarity
        )

        if not results:
            return None

        context_parts = []
        for i, (text, metadata, score) in enumerate(results, 1):
            source = metadata.get('filename', 'unknown')
            context_parts.append(f"[Источник {i}: {source} (релевантность: {score:.2f})]\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def send_message(self, user_input: str) -> str:
        self.add_user_message(user_input)
        api_messages = self.messages.copy()

        # --- RAG: получаем контекст и запоминаем результаты поиска ---
        rag_context = None
        rag_sources = set()
        if self.rag_enabled:
            query_vector = self.embedder.embed(user_input)
            results = self.vector_store.search(
                query_vector,
                top_k=self.top_k,
                min_similarity=self.min_similarity
            )
            if results:
                # Собираем контекст для промпта
                context_parts = []
                for i, (text, metadata, score) in enumerate(results, 1):
                    source = metadata.get('filename', 'unknown')
                    rag_sources.add(source)
                    context_parts.append(f"[Источник {i}: {source} (релевантность: {score:.2f})]\n{text}")
                rag_context = "\n\n---\n\n".join(context_parts)

        if rag_context:
            system_prompt = (
                "Ты - полезный ассистент. Используй следующий контекст из документов "
                "для ответа на вопрос. Если контекст не содержит ответа, "
                "скажи об этом и ответь на основе своих знаний.\n\n"
                f"КОНТЕКСТ ИЗ ДОКУМЕНТОВ:\n{rag_context}"
            )
            api_messages.insert(0, {"role": "system", "content": system_prompt})
            print(f"🔍 [RAG] Найдено релевантных фрагментов в: {', '.join(rag_sources)}")
        # -------------------------------------------------------------

        try:
            stream = ollama.chat(
                model=self.model,
                messages=api_messages,
                stream=True,
            )
            # Выбор префикса в зависимости от использования RAG
            if rag_context:
                print("\n🤖 [RAG] ", end="", flush=True)
            else:
                print("\n🤖 ", end="", flush=True)

            full_response = ""

            for chunk in stream:
                content = None
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    content = chunk.message.content
                elif isinstance(chunk, dict):
                    if 'message' in chunk and isinstance(chunk['message'], dict):
                        content = chunk['message'].get('content')
                    elif 'response' in chunk:
                        content = chunk['response']
                if content:
                    print(content, end="", flush=True)
                    full_response += content

            print("\n")
            self.add_assistant_message(full_response)

            # Вывод источников, если использовался RAG
            if rag_sources:
                print(f"📚 Использованные источники: {', '.join(rag_sources)}\n")

            return full_response

        except Exception as e:
            import traceback
            print("\n❌ Ошибка при отправке запроса:")
            traceback.print_exc()
            if self.messages and self.messages[-1]['role'] == 'user':
                self.messages.pop()
            return f"Ошибка: {e}"

    def clear_context(self) -> None:
        self.messages.clear()
        print("🧹 Контекст диалога очищен.\n")

    def show_context_length(self) -> None:
        count = len(self.messages)
        print(f"📚 Сообщений в контексте: {count}\n")

    def show_rag_stats(self) -> None:
        if not self.rag_enabled or not self.vector_store:
            print("ℹ️  RAG не активирован.\n")
            return
        stats = self.vector_store.get_stats()
        embed_stats = self.embedder.get_stats()
        print("\n📊 RAG Статистика:")
        print(f"  Документов в индексе: {len(stats['sources'])}")
        print(f"  Всего чанков: {stats['total_chunks']}")
        print(f"  Размерность векторов: {stats['vector_dimension']}")
        print(f"  Память: {stats['memory_usage_mb']:.2f} MB")
        print(f"  Кэш эмбеддингов: {embed_stats['cache_hits']} попаданий, {embed_stats['cache_misses']} промахов")
        print()

    def add_to_rag(self, path_str: str) -> None:
        """Добавляет файл или папку в индекс и сохраняет."""
        if not self.rag_enabled:
            print("❌ RAG не активирован. Сначала выполните /rag init")
            return

        path = Path(path_str)
        if not path.exists():
            print(f"❌ Путь не найден: {path_str}")
            return

        loader = DocumentLoader()
        chunks = []
        if path.is_file():
            try:
                chunks = loader.load_file(str(path))
            except Exception as e:
                print(f"❌ Ошибка загрузки файла: {e}")
                return
        else:
            chunks = loader.load_directory(str(path))

        if not chunks:
            print("⚠️ Не найдено подходящих документов.")
            return

        print(f"📄 Индексация {len(chunks)} чанков...")
        texts = [c.text for c in chunks]
        metadata = [c.metadata for c in chunks]

        vectors = self.embedder.embed_batch(texts)
        self.vector_store.add(texts, vectors, metadata)
        self.vector_store.save(RAG_INDEX_NAME)
        print(f"✅ Добавлено {len(chunks)} чанков. Индекс сохранён.")

    def list_rag_sources(self) -> None:
        if not self.rag_enabled or not self.vector_store:
            print("ℹ️ RAG не активирован.")
            return
        sources = self.vector_store.get_sources()
        if not sources:
            print("📭 Индекс пуст.")
        else:
            print("📚 Источники в индексе:")
            for src in sources:
                print(f"  • {Path(src).name}")

    def delete_from_rag(self, source_pattern: str) -> None:
        if not self.rag_enabled or not self.vector_store:
            print("ℹ️ RAG не активирован.")
            return
        sources = self.vector_store.get_sources()
        matched = [s for s in sources if source_pattern in s]
        if not matched:
            print(f"❌ Источник '{source_pattern}' не найден.")
            return
        if len(matched) > 1:
            print("Найдено несколько источников, уточните:")
            for m in matched:
                print(f"  • {m}")
            return
        deleted = self.vector_store.delete_by_source(matched[0])
        if deleted > 0:
            self.vector_store.save(RAG_INDEX_NAME)
            print(f"✅ Удалено {deleted} чанков из источника '{Path(matched[0]).name}'.")
        else:
            print("❌ Не удалось удалить.")

    def search_rag(self, query: str) -> None:
        if not self.rag_enabled or not self.vector_store:
            print("ℹ️ RAG не активирован.")
            return
        query_vector = self.embedder.embed(query)
        results = self.vector_store.search(query_vector, top_k=self.top_k)
        if not results:
            print("🔍 Ничего не найдено.")
            return
        print(f"\n🔍 Результаты поиска по запросу: '{query}'\n")
        for i, (text, meta, score) in enumerate(results, 1):
            source = Path(meta.get('source', '?')).name
            print(f"{i}. [{source}] (score: {score:.3f})")
            print(f"   {text[:200]}...\n")


def print_welcome(model: str, rag_enabled: bool):
    print("=" * 60)
    print("🤖 Локальный LLM Чат через Ollama")
    if rag_enabled:
        print("📚 Режим RAG активирован")
    print("=" * 60)
    print(f"Модель: {model}")
    print("\nСпециальные команды:")
    print("  /clear     - очистить контекст")
    print("  /count     - показать размер контекста")
    if rag_enabled:
        print("  /rag init [path]  - инициализировать RAG с хранилищем")
        print("  /rag add <path>   - добавить файл/папку в индекс")
        print("  /rag list         - показать источники в индексе")
        print("  /rag stats        - статистика индекса")
        print("  /rag delete <src> - удалить источник")
        print("  /rag search <q>   - поиск по индексу")
    else:
        print("  /rag init [path]  - включить RAG и создать индекс")
    print("  /exit      - выход")
    print("  /help      - помощь")
    print("\nВведите ваш вопрос:\n")


def print_help(rag_enabled: bool):
    print("\n📋 Доступные команды:")
    print("  /clear     - очистить контекст диалога")
    print("  /count     - показать количество сообщений в истории")
    if rag_enabled:
        print("  /rag init [path]  - инициализировать RAG с хранилищем")
        print("  /rag add <path>   - добавить файл/папку в индекс")
        print("  /rag list         - показать источники в индексе")
        print("  /rag stats        - статистика индекса")
        print("  /rag delete <src> - удалить источник")
        print("  /rag search <q>   - поиск по индексу")
    else:
        print("  /rag init [path]  - включить RAG и создать индекс")
    print("  /exit      - выйти из чата")
    print("  /help      - показать эту справку\n")


def handle_special_command(command: str, chat: LocalLLMChat) -> bool:
    parts = command.split(maxsplit=2)
    cmd = parts[0].lower()

    if cmd == "/exit":
        print("👋 До свидания!")
        return False
    elif cmd == "/clear":
        chat.clear_context()
    elif cmd == "/count":
        chat.show_context_length()
    elif cmd == "/rag":
        if len(parts) < 2:
            print("❓ Используйте: /rag [init|add|list|stats|delete|search]")
            return True
        subcmd = parts[1].lower()

        if subcmd == "init":
            path = parts[2] if len(parts) > 2 else str(RAG_DEFAULT_DIR)
            chat._init_rag(path, chat.embedder.model_name if chat.embedder else "nomic-embed-text")
            print(f"✅ RAG инициализирован с хранилищем: {path}")
        elif subcmd == "add":
            if len(parts) < 3:
                print("❓ Укажите путь к файлу или папке: /rag add <path>")
                return True
            chat.add_to_rag(parts[2])
        elif subcmd == "list":
            chat.list_rag_sources()
        elif subcmd == "stats":
            chat.show_rag_stats()
        elif subcmd == "delete":
            if len(parts) < 3:
                print("❓ Укажите источник для удаления: /rag delete <source>")
                return True
            chat.delete_from_rag(parts[2])
        elif subcmd == "search":
            if len(parts) < 3:
                print("❓ Укажите поисковый запрос: /rag search <query>")
                return True
            chat.search_rag(parts[2])
        else:
            print(f"❓ Неизвестная подкоманда: {subcmd}")
    elif cmd == "/help":
        print_help(chat.rag_enabled)
    else:
        print(f"❓ Неизвестная команда: {command}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI-чат с локальной LLM через Ollama (с поддержкой RAG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python chat.py --model llama3.2:3b
  python chat.py --model llama3.2:3b --no-rag
        """
    )
    parser.add_argument("-m", "--model", type=str, default="llama3.2:3b",
                        help="Название модели в Ollama")
    parser.add_argument("--no-rag", action="store_true",
                        help="Отключить RAG при старте")
    parser.add_argument("--embed-model", type=str, default="nomic-embed-text",
                        help="Модель для эмбеддингов")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Количество релевантных чанков")
    parser.add_argument("--min-similarity", type=float, default=0.3,
                        help="Минимальное сходство")
    return parser.parse_args()


def main():
    args = parse_args()

    # Проверяем, запущена ли Ollama
    try:
        ollama.list()
    except Exception:
        print("❌ Ошибка: Не удалось подключиться к Ollama.")
        print("Убедитесь, что:")
        print("  1. Ollama установлена")
        print("  2. Сервис Ollama запущен")
        sys.exit(1)

    chat = LocalLLMChat(
        model=args.model,
        no_rag=args.no_rag,
        embed_model=args.embed_model,
        top_k=args.top_k,
        min_similarity=args.min_similarity
    )

    print_welcome(chat.model, chat.rag_enabled)

    try:
        while True:
            try:
                user_input = input("👤 Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 До свидания!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not handle_special_command(user_input, chat):
                    break
                continue

            chat.send_message(user_input)

    except KeyboardInterrupt:
        print("\n\n👋 До свидания!")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()