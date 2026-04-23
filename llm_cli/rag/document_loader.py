"""Загрузчик документов с разбиением на чанки."""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass


@dataclass
class Chunk:
    """Структура для хранения чанка текста."""
    text: str
    metadata: Dict[str, Any]
    chunk_index: int
    total_chunks: int


class DocumentLoader:
    """
    Загрузчик документов с поддержкой различных форматов.
    Разбивает текст на перекрывающиеся чанки.
    """

    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.rst', '.py', '.js', '.html', '.css'}

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        Инициализация загрузчика.

        Args:
            chunk_size: размер чанка в символах
            chunk_overlap: перекрытие между чанками в символах
            separators: разделители для разбиения текста (в порядке приоритета)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Разделители по умолчанию (от большего к меньшему)
        self.separators = separators or [
            "\n\n",    # Параграфы
            "\n",      # Строки
            ". ",      # Предложения
            "! ",      # Восклицательные предложения
            "? ",      # Вопросительные предложения
            "; ",      # Части предложений
            ", ",      # Фразы
            " ",       # Слова
            ""         # Символы
        ]

    def _split_text(self, text: str) -> List[str]:
        """
        Разбивает текст на чанки с учётом перекрытия.

        Args:
            text: исходный текст

        Returns:
            список чанков
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            # Определяем конец чанка
            end = start + self.chunk_size

            if end >= len(text):
                # Последний чанк
                chunk = text[start:]
                if chunk.strip():
                    chunks.append(chunk)
                break

            # Ищем лучший разделитель для конца чанка
            chunk_text = text[start:end]

            # Пытаемся найти разделитель в порядке приоритета
            best_split = end
            for separator in self.separators:
                if not separator:
                    # Если разделитель пустой - разбиваем по символам
                    best_split = end
                    break

                # Ищем последнее вхождение разделителя в чанке
                last_sep = chunk_text.rfind(separator)
                if last_sep != -1:
                    # Нашли разделитель
                    potential_split = start + last_sep + len(separator)

                    # Не отступаем слишком далеко назад (максимум 20% размера чанка)
                    if end - potential_split < self.chunk_size * 0.2:
                        best_split = potential_split
                        break

            # Создаём чанк
            chunk = text[start:best_split]
            if chunk.strip():
                chunks.append(chunk)

            # Следующий чанк начинается с учётом перекрытия
            start = best_split - self.chunk_overlap
            if start < 0:
                start = 0

            # Избегаем бесконечного цикла
            if start >= best_split:
                start = best_split

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Очищает текст от лишних пробелов и нормализует.

        Args:
            text: исходный текст

        Returns:
            очищенный текст
        """
        # Заменяем множественные пробелы и переносы
        text = re.sub(r'\s+', ' ', text)
        # Убираем пробелы в начале и конце
        text = text.strip()
        return text

    def load_file(self, file_path: str) -> List[Chunk]:
        """
        Загружает и разбивает один файл на чанки.

        Args:
            file_path: путь к файлу

        Returns:
            список чанков
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Неподдерживаемый формат файла: {file_path.suffix}. "
                f"Поддерживаются: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        # Читаем файл
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Пробуем другие кодировки
            for encoding in ['cp1251', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Не удалось определить кодировку файла: {file_path}")

        # Очищаем текст
        content = self._clean_text(content)

        if not content:
            return []

        # Разбиваем на чанки
        chunk_texts = self._split_text(content)

        # Создаём объекты Chunk с метаданными
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    'source': str(file_path.absolute()),
                    'filename': file_path.name,
                    'file_type': file_path.suffix,
                    'chunk_index': i,
                    'total_chunks': len(chunk_texts),
                    'char_start': 0,  # Можно добавить точные позиции
                    'char_end': len(chunk_text)
                },
                chunk_index=i,
                total_chunks=len(chunk_texts)
            )
            chunks.append(chunk)

        return chunks

    def load_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        file_pattern: str = "*"
    ) -> List[Chunk]:
        """
        Загружает все поддерживаемые файлы из директории.

        Args:
            dir_path: путь к директории
            recursive: рекурсивно обходить поддиректории
            file_pattern: паттерн для файлов (например, "*.txt")

        Returns:
            список чанков из всех файлов
        """
        dir_path = Path(dir_path)

        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Директория не найдена: {dir_path}")

        all_chunks = []
        files_processed = 0

        # Собираем все подходящие файлы
        pattern = f"**/{file_pattern}" if recursive else file_pattern
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    chunks = self.load_file(str(file_path))
                    all_chunks.extend(chunks)
                    files_processed += 1
                    print(f"  ✓ {file_path.name}: {len(chunks)} чанков")
                except Exception as e:
                    print(f"  ✗ {file_path.name}: {e}")

        print(f"\n📊 Всего обработано файлов: {files_processed}")
        print(f"📊 Всего создано чанков: {len(all_chunks)}")

        return all_chunks

    def chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        """Преобразует Chunk в словарь."""
        return {
            'text': chunk.text,
            'metadata': chunk.metadata,
            'chunk_index': chunk.chunk_index,
            'total_chunks': chunk.total_chunks
        }


# Пример использования
if __name__ == "__main__":
    print("🔍 Тестирование DocumentLoader")
    print("=" * 50)

    # Создаём тестовый файл
    test_file = Path("./data/test_document.txt")
    test_file.parent.mkdir(exist_ok=True)

    test_content = """
    Python - это мощный язык программирования.

    Он используется для веб-разработки, анализа данных, 
    искусственного интеллекта и многого другого.

    NumPy - библиотека для научных вычислений.
    Pandas - библиотека для анализа данных.
    Ollama - инструмент для запуска LLM локально.

    Векторные базы данных позволяют эффективно 
    хранить и искать эмбеддинги текстов.
    """

    test_file.write_text(test_content, encoding='utf-8')

    # Тестируем загрузчик
    loader = DocumentLoader(chunk_size=100, chunk_overlap=20)

    print(f"\n📄 Загрузка файла: {test_file.name}")
    chunks = loader.load_file(str(test_file))

    print(f"\n📋 Создано {len(chunks)} чанков:")
    for i, chunk in enumerate(chunks):
        print(f"\n  Чанк {i + 1}/{len(chunks)}:")
        print(f"  Текст: {chunk.text[:80]}...")
        print(f"  Метаданные: источник={chunk.metadata['filename']}")

    # Тестируем загрузку директории
    print("\n📁 Тестирование загрузки директории:")
    dir_chunks = loader.load_directory("./data", recursive=False, file_pattern="*.txt")