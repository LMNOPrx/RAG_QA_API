import os
from dotenv import load_dotenv

# Load environment variables from .env file located at project root
load_dotenv()

class Settings:
    GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1")
    GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2")
    COHERE_API_KEY_1 = os.getenv("COHERE_API_KEY_1")
    COHERE_API_KEY_2 = os.getenv("COHERE_API_KEY_2")
    
    def __init__(self):
        # Optional: print or assert to catch missing env vars
        for key in ['GROQ_API_KEY_1', 'GROQ_API_KEY_2', 'COHERE_API_KEY_1', 'COHERE_API_KEY_2']:
            if not getattr(self, key):
                raise ValueError(f"{key} not found in environment.")

settings = Settings()
