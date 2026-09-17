from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Step 1: Load documents
documents = SimpleDirectoryReader("data").load_data()


# Step 2: Split documents into chunks
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=64
)

nodes = splitter.get_nodes_from_documents(documents)


# Step 3: Create embedding model
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 4: Convert first chunk into an embedding
embedding = embed_model.get_text_embedding(nodes[0].text)


# Step 5: Display embedding information
print("Number of chunks:", len(nodes))
print("Embedding vector length:", len(embedding))

print("\nFirst 10 numbers of the embedding:")
print(embedding[:10])