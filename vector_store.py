import chromadb

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext

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
# STEP 3: Create embedding model
# -----------------------------------------

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# -----------------------------------------
# STEP 4: Create ChromaDB
# -----------------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    "capstone_knowledge_base"
)

print("ChromaDB collection ready.")


# -----------------------------------------
# STEP 5: Connect ChromaDB with LlamaIndex
# -----------------------------------------

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# -----------------------------------------
# STEP 6: Create Vector Store Index
# -----------------------------------------

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model
)

print("Vector index created successfully.")


# -----------------------------------------
# STEP 7: Check stored data
# -----------------------------------------

print("Number of items in ChromaDB:",
      collection.count())