import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup RAG
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)
vectorstore = QdrantVectorStore(client=client, collection_name="constitution_rag", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

PROMPT = PromptTemplate(
    template="""
You are a legal assistant answering questions about the Indian Constitution.
Use ONLY the context provided. If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question:
{question}

Provide a clear and accurate explanation.
""",
    input_variables=["context", "question"]
)

llm = ChatGroq(
    api_key=os.environ.get("API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(request: QueryRequest):
    result = rag_chain.invoke(request.query)
    return {"answer": result}

@app.get("/")
def root():
    return {"status": "Samvidhan AI backend is running!"}

# Run with: uvicorn server:app --reload
