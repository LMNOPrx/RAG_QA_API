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
from langchain_community.retrievers import BM25Retriever

from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableLambda
from functools import lru_cache

FAISS_INDEX_PATH = "./faiss_index"


def get_loader(file_path: str, file_extension: str):
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
        model="embed-english-v3.0",
        cohere_api_key=settings.COHERE_API_KEY_1,
        user_agent="rag-app",
    )


def extract_metadata(chunk: Document) -> dict:
    metadata = chunk.metadata or {}
    lines = chunk.page_content.split("\n")
    for line in lines:
        if line.lower().strip().startswith("section"):
            metadata["section"] = line.strip()
            break
    if "page" not in metadata:
        metadata["page"] = metadata.get("page_number", -1)
    return metadata


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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        for split in splits:
            split.metadata.update(extract_metadata(split))

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
    doc_hash = hashlib.sha256(doc_url.encode()).hexdigest()
    index_file = os.path.join(FAISS_INDEX_PATH, f"{doc_hash}.faiss")
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

    dense_retriever = vectorstore.as_retriever(search_kwargs={'k': 25})
    bm25_retriever = BM25Retriever.from_documents(vectorstore.docstore._dict.values())
    bm25_retriever.k = 25

    hybrid_retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.5, 0.5])

    compressor = CohereRerank(
        cohere_api_key=settings.COHERE_API_KEY_2,
        model="rerank-english-v3.0",
        top_n=5
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever
    )

    return compression_retriever


def multi_query_expansion(query: str) -> List[str]:
    expansion_prompt = ChatPromptTemplate.from_messages([
        ("system", "Expand the given query into multiple specific sub-questions that cover different aspects of the original."),
        ("human", "{query}")
    ])

    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.3, api_key=settings.GROQ_API_KEY_1)
    chain = expansion_prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": query})
    return [line.strip("-• ") for line in raw.split("\n") if line.strip()]


def docs2fid_prompt(docs, question):
    prompt = ""
    for i, doc in enumerate(docs):
        meta = doc.metadata
        header = f"[Section: {meta.get('section', 'Unknown')} | Page: {meta.get('page', '?')}]"
        prompt += f"Context {i+1} ({header}):\n{doc.page_content}\n\n"
    prompt += f"Question: {question}"
    return prompt


def invoke_rag_chain(doc_url: str, questions: List[str]) -> List[str]:
    retriever = get_retriever(doc_url)
    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.1, api_key=settings.GROQ_API_KEY_2)

    def generate_answer(inputs):
        docs = retriever.invoke(inputs)
        prompt = docs2fid_prompt(docs, inputs)
        raw_response = llm.invoke(prompt).content

        cleaned = (
            raw_response.replace("\n", " ")        
            .replace("•", "-")                     
            .strip()                               
        )

        return cleaned

    rag_chain = RunnableLambda(generate_answer)

    return rag_chain.batch(questions)
