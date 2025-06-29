# rag_chatbot.py

import os
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader

# Constants
PDF_FOLDER = "data"
INDEX_FOLDER = "faiss_index"
GROQ_API_KEY = "write your grok api key here"  
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Load or create FAISS index
def load_faiss_index():
    if Path(INDEX_FOLDER).exists():
        return FAISS.load_local(INDEX_FOLDER, HuggingFaceEmbeddings(model_name=EMBED_MODEL), allow_dangerous_deserialization=True)
    else:
        docs = []
        for file in Path(PDF_FOLDER).glob("*.pdf"):
            loader = PyMuPDFLoader(str(file))
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(INDEX_FOLDER)
        return db


# Query function
def query_rag(query, model_name="llama-3.3-70b-versatile"):
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    db = load_faiss_index()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(temperature=0.2, model_name=model_name, api_key=GROQ_API_KEY)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    result = qa_chain.run(query)
    return result
