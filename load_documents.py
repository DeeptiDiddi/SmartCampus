from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()

print("Number of documents:", len(documents))

for document in documents:
    print("--------------------------------")
    print("File:", document.metadata.get("file_name"))
    print(document.text[:300])