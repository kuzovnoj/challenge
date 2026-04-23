import ollama
import json

print("Вызов ollama.list()")
response = ollama.list()

print("\nТип ответа:", type(response))
print("\nСырой ответ:")
print(json.dumps(response, indent=2, default=str))