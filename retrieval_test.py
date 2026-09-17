import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext
)

from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.vector_stores.chroma import ChromaVectorStore


# -----------------------------------------
# STEP 1: Load documents
# -----------------------------------------

documents = SimpleDirectoryReader("data").load_data()

print("Documents loaded:", len(documents))


# -----------------------------------------
# STEP 2: Chunk documents
# -----------------------------------------

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=64
)

nodes = splitter.get_nodes_from_documents(documents)

print("Chunks created:", len(nodes))


# -----------------------------------------
# STEP 3: Embedding model
# -----------------------------------------

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# -----------------------------------------
# STEP 4: Connect to ChromaDB
# -----------------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    "capstone_knowledge_base"
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# -----------------------------------------
# STEP 5: Create index
# -----------------------------------------

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)


# -----------------------------------------
# STEP 6: Create retriever
# -----------------------------------------

retriever = index.as_retriever(
    similarity_top_k=3
)


# -----------------------------------------
# STEP 7: Test question
# -----------------------------------------

question = "What is the minimum attendance required for ESE?"

print("\nQuestion:")
print(question)

print("\nRetrieved Chunks:")
print("--------------------------------")


results = retriever.retrieve(question)


for i, result in enumerate(results):

    print(f"\nResult {i + 1}")
    print("Similarity Score:", result.score)
    print("Source:", result.node.metadata.get("file_name"))

    print("\nContent:")
    print(result.node.text)