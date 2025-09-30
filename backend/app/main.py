# main.py

# --- Standard Library and Environment Imports ---
import os
import asyncio
import logging
from dotenv import load_dotenv

# This MUST be one of the first lines to ensure environment variables are loaded.
load_dotenv()

# --- Third-Party Imports ---
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import security and performance middleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Your Application's Imports ---
from .readmode import router as read_router
from .engine import engine

# --- Application Setup ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

# --- Pydantic Model for Input Validation ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=300)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Gita AI API",
    description="An API for querying and reading the Bhagavad Gita.",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS Middleware Configuration ---
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")
if not ALLOWED_ORIGINS or not ALLOWED_ORIGINS[0]:
    logger.warning("ALLOWED_ORIGINS environment variable not set. Defaulting to 'http://localhost:3000'.")
    ALLOWED_ORIGINS = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Static Files and Routers ---
app.mount("/static", StaticFiles(directory="data/audio"), name="static")
app.include_router(read_router, prefix="/api")

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"status": "Gita AI API is running."}


@app.post("/api/ask")
@limiter.limit("50/minute")
# ▼▼▼▼▼ THE FIX IS HERE ▼▼▼▼▼
async def ask_question(query_body: QueryRequest, request: Request):
# ▲▲▲▲▲ THE FIX IS HERE ▲▲▲▲▲
    """
    Asynchronously handles a user's question with security and performance in mind.
    """
    try:
        # Use query_body.query to access the user's input
        response_data = await asyncio.to_thread(engine.get_response, query_body.query)
        return response_data
    except Exception as e:
        logger.error(f"Engine error for query '{query_body.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later."
        )