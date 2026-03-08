import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 1. Load PDFs
documents = []
folder_path = "D:\\zzzzz\\constitution"

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder_path, file))
        docs = loader.load()
        documents.extend(docs)

print("Total pages loaded:", len(documents))

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)
print("Total chunks:", len(chunks))

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create collection in Qdrant manually
client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="constitution_rag",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 5. Store chunks
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="constitution_rag",
    embedding=embeddings
)

vectorstore.add_documents(chunks)

print("Vector database created successfully!")