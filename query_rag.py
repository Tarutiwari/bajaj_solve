# from langchain_qdrant import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(host="localhost", port=6333)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="constitution_rag",
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

query = "What does Article 14 say?"

docs = retriever.invoke(query)

for doc in docs:
    print(doc.page_content)
    print("\n-------------------\n")