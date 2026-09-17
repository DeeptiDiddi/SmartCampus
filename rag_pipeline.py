import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama


# ==================================================
# 1. CONNECT TO CHROMADB
# ==================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    "capstone_knowledge_base"
)

print("ChromaDB loaded.")
print("Number of chunks:", collection.count())


# ==================================================
# 2. CONNECT CHROMADB TO LLAMAINDEX
# ==================================================

vector_store = ChromaVectorStore(
    chroma_collection=collection
)


# ==================================================
# 3. LOAD EMBEDDING MODEL
# ==================================================

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ==================================================
# 4. LOAD EXISTING VECTOR INDEX
# ==================================================

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model
)

print("Vector index loaded.")


# ==================================================
# 5. CREATE RETRIEVER
# ==================================================

retriever = index.as_retriever(
    similarity_top_k=3
)


# ==================================================
# 6. CONNECT TO OLLAMA
# ==================================================

llm = Ollama(
    model="qwen3.5:2b",
    request_timeout=300.0
)

print("Ollama connected.")


# ==================================================
# 7. ASK STUDENT QUESTION
# ==================================================

question = input(
    "\nAsk your Smart Campus question: "
)


# ==================================================
# 8. RETRIEVE RELEVANT CHUNKS
# ==================================================

results = retriever.retrieve(question)

print("\nRelevant information retrieved.")


# ==================================================
# 9. FILTER IRRELEVANT RESULTS
# ==================================================

filtered_results = [
    result
    for result in results
    if result.score >= 0.35
]


# ==================================================
# 10. DISPLAY FILTERED RESULTS
# ==================================================

print("\nFiltered Results:")

for i, result in enumerate(filtered_results):

    print("\n--------------------------------")
    print("Result:", i + 1)
    print("Similarity Score:", result.score)
    print(
        "Source:",
        result.node.metadata.get("file_name")
    )
    print("--------------------------------")

    print(result.node.text[:500])


# ==================================================
# 11. CREATE CONTEXT
# ==================================================

context = ""

for i, result in enumerate(filtered_results):

    context += f"""
Source: {result.node.metadata.get("file_name")}

Content:
{result.node.text}

"""


# ==================================================
# 12. CREATE GROUNDED PROMPT
# ==================================================

prompt = f"""
You are Smart Campus Assistant.

Answer the student's question ONLY using the
information provided in the context below.

Do not use outside knowledge.

If the answer is not available in the context,
say exactly:

"I don't have enough information to answer that."

Give a clear and concise answer.

Context:
{context}

Student Question:
{question}

Answer:
"""


# ==================================================
# 13. SEND PROMPT TO OLLAMA
# ==================================================

response = llm.complete(prompt)


# ==================================================
# 14. DISPLAY FINAL ANSWER
# ==================================================

print("\n==============================")
print("Smart Campus Assistant")
print("==============================")

print("\nAnswer:")
print(response)


# ==================================================
# 15. DISPLAY ONLY FILTERED SOURCES
# ==================================================

print("\nSources:")

for result in filtered_results:

    print(
        "-",
        result.node.metadata.get("file_name")
    )