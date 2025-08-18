from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.retrieval_pipeline import Badal
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AIRAC API", version="1.0.0")

# Enable CORS for React frontend (updated for Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Your Vite port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# Initialize the RAG pipeline once when the server starts
try:
    badal_pipeline = Badal()
    logger.info("RAG pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG pipeline: {e}")
    badal_pipeline = None

@app.get("/")
async def root():
    return {"message": "AIRAC API is running"}

@app.get("/health")
async def health_check():
    pipeline_status = "operational" if badal_pipeline else "failed"
    return {
        "status": "healthy", 
        "message": "API is operational",
        "rag_pipeline": pipeline_status
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        # Check if RAG pipeline is available
        if badal_pipeline is None:
            raise HTTPException(
                status_code=503, 
                detail="RAG pipeline is not available. Please check server logs."
            )
        
        # Validate input
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400, 
                detail="Query cannot be empty"
            )
        
        logger.info(f"Processing query: {request.query}")
        
        # Use your existing RAG pipeline
        ai_response = badal_pipeline.invoke(request.query.strip())
        
        logger.info("Query processed successfully")
        return QueryResponse(response=ai_response)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {str(e)}")
        
        # Return a user-friendly error message
        error_message = (
            "I encountered an issue processing your query. "
            "This could be due to connectivity issues with the knowledge base or AI service. "
            "Please try again in a moment."
        )
        return QueryResponse(response=error_message)