# from qdrant_client import QdrantClient

# client = QdrantClient(host="localhost", port=6333)

# collections = client.get_collections()
# print(collections)

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def ask(query):
    docs = retriever.invoke(query)
    
    print("=" * 60)
    print(f"Question: {query}")
    print("=" * 60)
    print("Relevant Information Found:\n")
    
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.page_content[:400]}")
        print()

# Test karo
ask("What are Fundamental Rights in Indian Constitution?")
ask("What does Article 21 say?")
ask("What is mentioned in Third Schedule?")