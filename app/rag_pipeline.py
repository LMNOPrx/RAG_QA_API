import requests
import os
import tempfile
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
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from functools import lru_cache

from app.schemas import QueryDetails

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


# Cache client and embedding initialization
@lru_cache(maxsize=1)
def get_pinecone_client():
    return PineconeClient(api_key=settings.PINECONE_API_KEY)

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

    pinecone_client = get_pinecone_client()
    index_name = "lmnoprx-rag"
    embeddings = get_embedding_model()
    
    namespace = hashlib.sha256(doc_url.encode()).hexdigest()[:32]
    
    index = pinecone_client.Index(index_name)
    existing_namespaces = index.describe_index_stats().get("namespaces", {})

    if namespace not in existing_namespaces:
        PineconeVectorStore.from_documents(
            documents=get_chunks(doc_url), 
            embedding=embeddings, 
            index_name=index_name,
            namespace=namespace
        )

    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    ).as_retriever(search_kwargs={'namespace': namespace, 'k': 10})


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

