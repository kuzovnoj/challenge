#!/usr/bin/env python3
"""
Простое CLI-приложение для общения с локальной LLM через Ollama.
Использование: python chat.py --model llama3.2
"""

import argparse
import sys
from typing import List, Dict

try:
    import ollama
except ImportError:
    print("Ошибка: библиотека ollama не установлена.")
    print("Установите её командой: pip install ollama")
    sys.exit(1)


class LocalLLMChat:
    """Класс для управления чатом с локальной LLM."""

    def __init__(self, model: str):
        """
        Инициализация чата.

        Args:
            model: название модели в Ollama (например, llama3.2, mistral)
        """
        self.model = model
        self.messages: List[Dict[str, str]] = []
        self._check_model_availability()

    def _check_model_availability(self) -> None:
        """Проверяет, доступна ли указанная модель в Ollama."""
        try:
            models = ollama.list()
            available_models = [m['name'] for m in models.get('models', [])]
            # Проверяем точное совпадение или модель с тегом
            model_exists = any(
                self.model in m or m.startswith(f"{self.model}:")
                for m in available_models
            )
            if not model_exists:
                print(f"⚠️  Модель '{self.model}' не найдена локально.")
                print("Ollama попытается загрузить её автоматически при первом запросе.")
                print(f"Доступные модели: {', '.join(available_models) if available_models else 'нет'}")
                print()
        except Exception as e:
            print(f"⚠️  Не удалось проверить список моделей: {e}")
            print("Убедитесь, что Ollama запущена (команда: ollama serve)")
            print()

    def add_user_message(self, content: str) -> None:
        """Добавляет сообщение пользователя в историю."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Добавляет ответ модели в историю."""
        self.messages.append({"role": "assistant", "content": content})

    def send_message(self, user_input: str) -> str:
        """
        Отправляет сообщение модели и возвращает ответ со стримингом.

        Args:
            user_input: текст сообщения пользователя

        Returns:
            полный ответ модели
        """
        self.add_user_message(user_input)

        try:
            # Отправляем запрос с поддержкой стриминга
            stream = ollama.chat(
                model=self.model,
                messages=self.messages,
                stream=True,
            )

            print("\n🤖 ", end="", flush=True)
            full_response = ""

            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    full_response += content

            print("\n")
            self.add_assistant_message(full_response)
            return full_response

        except ollama.ResponseError as e:
            error_msg = f"Ошибка API Ollama: {e.error}"
            print(f"\n❌ {error_msg}")
            # Удаляем последнее сообщение пользователя из истории,
            # так как запрос не удался
            self.messages.pop()
            return error_msg
        except Exception as e:
            error_msg = f"Неожиданная ошибка: {e}"
            print(f"\n❌ {error_msg}")
            self.messages.pop()
            return error_msg

    def clear_context(self) -> None:
        """Очищает историю диалога."""
        self.messages.clear()
        print("🧹 Контекст диалога очищен.\n")

    def show_context_length(self) -> None:
        """Показывает количество сообщений в контексте."""
        count = len(self.messages)
        print(f"📚 Сообщений в контексте: {count}\n")


def parse_args():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="CLI-чат с локальной LLM через Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python chat.py --model llama3.2
  python chat.py --model mistral
  python chat.py -m deepseek-r1:7b

Специальные команды в чате:
  /clear  - очистить контекст диалога
  /count  - показать количество сообщений в контексте
  /exit   - выход из чата
  /help   - показать это сообщение
        """
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="llama3.2",
        help="Название модели в Ollama (по умолчанию: llama3.2)"
    )
    return parser.parse_args()


def print_welcome(model: str):
    """Выводит приветственное сообщение."""
    print("=" * 50)
    print("🤖 Локальный LLM Чат через Ollama")
    print("=" * 50)
    print(f"Модель: {model}")
    print("\nСпециальные команды:")
    print("  /clear  - очистить контекст")
    print("  /count  - показать размер контекста")
    print("  /exit   - выход")
    print("  /help   - помощь")
    print("\nВведите ваш вопрос:\n")


def handle_special_command(command: str, chat: LocalLLMChat) -> bool:
    """
    Обрабатывает специальные команды чата.

    Args:
        command: команда от пользователя
        chat: экземпляр чата

    Returns:
        True если нужно продолжить работу, False для выхода
    """
    if command == "/exit":
        print("👋 До свидания!")
        return False
    elif command == "/clear":
        chat.clear_context()
    elif command == "/count":
        chat.show_context_length()
    elif command == "/help":
        print("\n📋 Доступные команды:")
        print("  /clear  - очистить контекст диалога")
        print("  /count  - показать количество сообщений в истории")
        print("  /exit   - выйти из чата")
        print("  /help   - показать эту справку\n")
    else:
        print(f"❓ Неизвестная команда: {command}")
        print("   Используйте /help для списка команд\n")
    return True


def main():
    """Главная функция приложения."""
    args = parse_args()

    # Проверяем, запущена ли Ollama
    try:
        ollama.list()
    except Exception:
        print("❌ Ошибка: Не удалось подключиться к Ollama.")
        print("Убедитесь, что:")
        print("  1. Ollama установлена")
        print("  2. Сервис Ollama запущен (команда: ollama serve)")
        sys.exit(1)

    # Создаем экземпляр чата
    chat = LocalLLMChat(args.model)

    print_welcome(args.model)

    # Основной цикл чата
    try:
        while True:
            try:
                user_input = input("👤 Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 До свидания!")
                break

            if not user_input:
                continue

            # Проверяем специальные команды
            if user_input.startswith("/"):
                if not handle_special_command(user_input, chat):
                    break
                continue

            # Отправляем обычное сообщение
            chat.send_message(user_input)

    except KeyboardInterrupt:
        print("\n\n👋 До свидания!")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()