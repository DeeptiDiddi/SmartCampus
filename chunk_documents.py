from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter


# Step 1: Load documents
documents = SimpleDirectoryReader("data").load_data()

print("Number of documents:", len(documents))


# Step 2: Create chunking strategy
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=64
)


# Step 3: Split documents into chunks
nodes = splitter.get_nodes_from_documents(documents)


# Step 4: Display results
print("Number of chunks:", len(nodes))

for i, node in enumerate(nodes[:5]):
    print("\n--------------------------------")
    print("Chunk:", i + 1)
    print("Source:", node.metadata.get("file_name"))
    print("--------------------------------")
    print(node.text)