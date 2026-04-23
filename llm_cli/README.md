# LLM CLI Chat с RAG

Локальный чат с LLM через Ollama и поддержкой Retrieval-Augmented Generation.

## 🚀 Возможности

- ✅ Полностью локальная работа (без облака)
- ✅ Чат с любой моделью из Ollama
- ✅ RAG: поиск по вашим документам
- ✅ Стриминг ответов в реальном времени
- ✅ Сохранение контекста диалога
- ✅ Кэширование эмбеддингов для скорости
- ✅ Поддержка txt, md, py, js, html, css файлов

## 📦 Установка

```bash
# 1. Клонируйте репозиторий
git clone <repo>
cd llm-cli

# 2. Установите зависимости
pip install -r requirements.txt

# 3. Установите Ollama и модели
# Скачайте с https://ollama.ai
ollama pull llama3.2
ollama pull nomic-embed-text