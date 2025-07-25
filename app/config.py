import os
from dotenv import load_dotenv

# Load environment variables from .env file located at project root
load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    def __init__(self):
        # Optional: print or assert to catch missing env vars
        for key in ['GOOGLE_API_KEY', 'PINECONE_API_KEY', 'LANGCHAIN_API_KEY']:
            if not getattr(self, key):
                raise ValueError(f"{key} not found in environment.")

settings = Settings()
