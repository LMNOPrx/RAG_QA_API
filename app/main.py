from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import uvicorn
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


from app.rag_pipeline import invoke_rag_chain  # Your RAG logic

# === FastAPI app with versioned API ===
app = FastAPI(
    title="HackRx Document QA API",
    version="1.0.0",
)

# === Auth Token ===
EXPECTED_TOKEN = "2df1ab83345b896947425c6ae8c9beeb25eb50b0e4e7b731e7e70c4783feb456"

# === Request & Response Schemas ===
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# === Auth Dependency ===
security = HTTPBearer(auto_error=False)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None or not credentials.scheme.lower() == "bearer":
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

    token = credentials.credentials
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    return True

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# === Endpoint ===
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx_rag(
    payload: HackRxRequest,
    auth: bool = Depends(verify_token)
):
    try:
        doc_url = str(payload.documents)
        answers = invoke_rag_chain(doc_url, payload.questions)
        return HackRxResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing RAG chain: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
