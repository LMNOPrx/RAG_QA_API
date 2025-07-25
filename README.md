# How to run (Local run).
### 1. In your rag_hackrx make .env file, inside it:
```.env
PINECONE_API_KEY = "<your-pinecone-api-key"
# right now it is using llama3:8b via ollama (running locally on machine) but if you want different LLM, say gpt-4 then write your OPENAI_API_KEY here.
```
```terminal
#make sure you are in the rag_hackrx directory (not in app dir)
uvicorn app.main:app --reload
```
then just go to the url `http://localhost:8000/`
