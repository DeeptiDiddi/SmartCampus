from llama_index.llms.ollama import Ollama

# Connect to local Ollama model
llm = Ollama(
    model="qwen3.5:2b",
    request_timeout=300.0
)

print("Sending question to Ollama...")

response = llm.complete(
    "Explain RAG in one simple sentence."
)

print("\nLLM Response:")
print(response)