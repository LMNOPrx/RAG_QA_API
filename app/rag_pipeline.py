import requests
import os
import tempfile
import time
from typing import List
from urllib.parse import urlparse
import hashlib

from app.config import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader,
)
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from functools import lru_cache

FAISS_INDEX_PATH = "./faiss_index"

def get_loader(file_path: str, file_extension: str):
    """Selects the appropriate document loader based on the file extension."""
    if file_extension == ".pdf":
        return PyPDFLoader(file_path)
    elif file_extension == ".docx":
        return UnstructuredWordDocumentLoader(file_path)
    elif file_extension in [".eml", ".msg"]:
        return UnstructuredEmailLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

@lru_cache(maxsize=1)
def get_embedding_model():
    return CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=settings.COHERE_API_KEY,
        user_agent="rag-app",
    )

def get_chunks(doc_url: str) -> List[Document]:

    temp_dir = tempfile.mkdtemp()
    try:
        response = requests.get(doc_url)
        response.raise_for_status()
        parsed_url = urlparse(doc_url)
        file_name = os.path.basename(parsed_url.path) or "document"
        _, file_extension = os.path.splitext(file_name)
        if not file_extension:
            raise ValueError("Could not determine file type from URL.")
        
        temp_file_path = os.path.join(temp_dir, file_name)
        with open(temp_file_path, "wb") as f:
            f.write(response.content)

        loader = get_loader(temp_file_path, file_extension)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"Failed to download or identify the document: {e}")
    finally:
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            os.rmdir(temp_dir)
        except Exception as cleanup_error:
            print(f"Cleanup failed: {cleanup_error}")
    
    return splits


def get_retriever(doc_url: str) -> Runnable:

    embeddings = get_embedding_model()
    
    # Create a unique, safe filename for the local index based on the document URL
    doc_hash = hashlib.sha256(doc_url.encode()).hexdigest()
    index_file = os.path.join(FAISS_INDEX_PATH, f"{doc_hash}.faiss")

    # Create the directory for indexes if it doesn't exist
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    if os.path.exists(index_file):
        print(f"INFO: FAISS index found for document. Loading from: {index_file}")
        vectorstore = FAISS.load_local(index_file, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"INFO: No FAISS index found. Processing and creating new index for document.")
        chunks = get_chunks(doc_url)
        if not chunks:
            raise ValueError("Document processing yielded no chunks.")
        
        print("INFO: Creating FAISS index from document chunks...")
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        
        print(f"INFO: Saving FAISS index to: {index_file}")
        vectorstore.save_local(index_file)

    return vectorstore.as_retriever(search_kwargs={'k': 10})


def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def invoke_rag_chain(doc_url: str, questions: List[str]) -> List[str]:

    retriever = get_retriever(doc_url)

    model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.5)

    prompt_template = ChatPromptTemplate([
        (
            "system",

            """
            You are a helpful assistant trained to answer queries from complex documents (eg. ensurance policiy document).
            Use only the following context to answer the user's question precisely, constructive and in one or two sentence. 
            Answer in a concise and complete manner. the answer should be complete (all the relevant information should be in the answer, don't tell user to look up something in the document or say something like 'in the given table.').
            
            """
        ),
        (
            "human",
            """
            query:
            {query}

            context:
            {context}
            """
        )
    ])

    rag_chain = (
        {"context": retriever | docs2str, "query": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )


    answers = rag_chain.batch(questions)
    return answers

