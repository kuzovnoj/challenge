#!/usr/bin/env python3
"""
Веб-интерфейс для локального LLM-чата с RAG и оптимизацией.
Использует Gradio для веб-доступа.
"""

import gradio as gr
import time
from pathlib import Path
from chat import LocalLLMChat, create_chat_instance

# Глобальные переменные
chat_instance: LocalLLMChat = None
metrics_history = []

def init_chat(model: str, enable_rag: bool, embed_model: str):
    """Инициализирует или пересоздаёт чат с новыми параметрами."""
    global chat_instance
    chat_instance = LocalLLMChat(
        model=model,
        no_rag=not enable_rag,
        embed_model=embed_model
    )
    return "✅ Чат инициализирован. Можете начинать общение."

def chat_respond(message: str, history: list):
    """Обрабатывает сообщение пользователя и возвращает ответ с метриками."""
    global chat_instance, metrics_history
    
    if chat_instance is None:
        return "❌ Чат не инициализирован. Нажмите 'Инициализировать чат'."
    
    if not message.strip():
        return ""
    
    # Отправляем сообщение (метод send_message уже выводит метрики в терминал)
    # Но нам нужно перехватить метрики для отображения в вебе
    start_time = time.time()
    
    # Сохраняем старый stdout для перехвата
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        response = chat_instance.send_message(message)
        terminal_output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    
    end_time = time.time()
    
    # Парсим метрики из вывода (можно сделать красивее, но для примера так)
    metrics_text = ""
    if "Метрики:" in terminal_output:
        metrics_lines = terminal_output.split("Метрики:")[1].split("\n")
        metrics_text = "📊 **Метрики:**\n"
        for line in metrics_lines[:10]:
            if line.strip():
                metrics_text += f"> {line.strip()}\n"
    
    # Сохраняем в историю метрик
    metrics_history.append({
        "time": end_time - start_time,
        "message": message[:50],
        "timestamp": time.strftime("%H:%M:%S")
    })
    
    # Формируем ответ с метриками
    full_response = response
    if metrics_text:
        full_response += f"\n\n---\n{metrics_text}"
    
    return full_response

def toggle_rag(enable: bool):
    """Включает/выключает RAG."""
    if chat_instance is None:
        return "❌ Чат не инициализирован"
    
    if enable:
        if not chat_instance.rag_enabled:
            chat_instance._init_rag("./rag_data", "nomic-embed-text")
        return "✅ RAG включён"
    else:
        chat_instance.rag_enabled = False
        return "✅ RAG выключен"

def add_document(file_obj):
    """Добавляет загруженный файл в RAG."""
    if chat_instance is None or not chat_instance.rag_enabled:
        return "❌ RAG не активирован. Включите RAG и попробуйте снова."
    
    if file_obj is None:
        return "❌ Файл не выбран."
    
    # Сохраняем временный файл
    temp_path = Path(file_obj.name)
    
    try:
        chat_instance.add_to_rag(str(temp_path))
        return f"✅ Файл {temp_path.name} добавлен в RAG"
    except Exception as e:
        return f"❌ Ошибка: {e}"

def clear_context():
    """Очищает контекст диалога."""
    if chat_instance:
        chat_instance.clear_context()
        return "✅ Контекст очищен"
    return "❌ Чат не инициализирован"

def get_stats():
    """Возвращает статистику чата."""
    if chat_instance is None:
        return "Чат не инициализирован"
    
    stats = f"""
**Модель:** {chat_instance.model}
**RAG:** {'включён' if chat_instance.rag_enabled else 'выключен'}
**Контекст:** {len(chat_instance.messages)} сообщений
**Оптимизация:** {'включена' if chat_instance.optimize_mode else 'выключена'}
"""
    if chat_instance.rag_enabled and chat_instance.vector_store:
        rag_stats = chat_instance.vector_store.get_stats()
        stats += f"\n**Документов в RAG:** {len(rag_stats.get('sources', []))}"
        stats += f"\n**Чанков:** {rag_stats.get('total_chunks', 0)}"
    
    return stats

# Создаём интерфейс
with gr.Blocks(title="Локальный LLM Чат", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Локальный LLM Чат с RAG")
    
    # Боковая панель с настройками
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Настройки")
            
            model_input = gr.Dropdown(
                choices=["llama3.2:3b", "deepseek-r1:1.5b", "mistral"],
                value="llama3.2:3b",
                label="Модель"
            )
            embed_model_input = gr.Dropdown(
                choices=["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
                value="nomic-embed-text",
                label="Модель эмбеддингов"
            )
            rag_toggle = gr.Checkbox(label="Включить RAG", value=True)
            
            init_btn = gr.Button("🔄 Инициализировать чат", variant="primary")
            init_status = gr.Textbox(label="Статус", interactive=False)
            
            gr.Markdown("---")
            
            # RAG управление
            gr.Markdown("### RAG Управление")
            file_upload = gr.File(label="Загрузить документ")
            upload_btn = gr.Button("📄 Добавить в RAG")
            upload_status = gr.Textbox(label="Результат", interactive=False)
            
            gr.Markdown("---")
            
            # Управление чатом
            clear_btn = gr.Button("🧹 Очистить контекст")
            stats_btn = gr.Button("📊 Статистика")
            stats_output = gr.Markdown()
            
        # Основной чат
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat_respond,
                chatbot=gr.Chatbot(height=500),
                textbox=gr.Textbox(placeholder="Введите сообщение...", container=False),
                title="",
                description="",
            )
    
    # Обработчики событий
    init_btn.click(
        init_chat,
        inputs=[model_input, rag_toggle, embed_model_input],
        outputs=init_status
    )
    
    upload_btn.click(
        add_document,
        inputs=file_upload,
        outputs=upload_status
    )
    
    clear_btn.click(
        clear_context,
        outputs=stats_output
    )
    
    stats_btn.click(
        get_stats,
        outputs=stats_output
    )
    
    rag_toggle.change(
        toggle_rag,
        inputs=rag_toggle,
        outputs=init_status
    )

# Запуск
if __name__ == "__main__":
    # Доступно по http://localhost:7860
    # Установите share=True для публичной ссылки
    demo.launch(
        server_name="0.0.0.0",  # Доступ по локальной сети
        server_port=7860,
        share=True,  # Создаст публичную ссылку (действует 72ч)
        inbrowser=True  # Автоматически откроет браузер
    )