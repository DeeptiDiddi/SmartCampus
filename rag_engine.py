import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# --------------------------------------------------
# STEP 1: Load the existing ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

print("ChromaDB loaded.")


# --------------------------------------------------
# STEP 2: Get our existing collection
# --------------------------------------------------

collection = chroma_client.get_collection(
    "capstone_knowledge_base"
)

print("Collection loaded.")
print("Number of items:", collection.count())


# --------------------------------------------------
# STEP 3: Connect ChromaDB to LlamaIndex
# --------------------------------------------------

vector_store = ChromaVectorStore(
    chroma_collection=collection
)


# --------------------------------------------------
# STEP 4: Load the same embedding model
# --------------------------------------------------

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# STEP 5: Create index from EXISTING vector store
# --------------------------------------------------

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model
)

print("Existing vector index loaded.")


# --------------------------------------------------
# STEP 6: Create retriever
# --------------------------------------------------

retriever = index.as_retriever(
    similarity_top_k=3
)


# --------------------------------------------------
# STEP 7: Ask a question
# --------------------------------------------------

question = "What is the minimum attendance required for ESE?"

print("\nQuestion:")
print(question)


# --------------------------------------------------
# STEP 8: Retrieve relevant chunks
# --------------------------------------------------

results = retriever.retrieve(question)


# --------------------------------------------------
# STEP 9: Display results
# --------------------------------------------------

print("\nRetrieved Results:")

for i, result in enumerate(results):

    print("\n--------------------------------")
    print("Result:", i + 1)
    print("Similarity Score:", result.score)
    print("Source:", result.node.metadata.get("file_name"))
    print("--------------------------------")

    print(result.node.text)