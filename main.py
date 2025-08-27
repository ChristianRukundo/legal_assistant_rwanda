import os
import asyncio
import uuid
import time
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from pathlib import Path
import aiofiles
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from advanced_rag_pipeline import AdvancedRAGPipeline
from sophisticated_voice_processor import SophisticatedVoiceProcessor
from enterprise_caching_system import enterprise_caching_system
from intelligent_query_processor import ComprehensiveQueryAnalyzer
from advanced_document_processor import DocumentIntelligenceProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components
rag_pipeline = None
voice_processor = None
cache_manager = None
query_analyzer = None
document_processor = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    language: str = "rw"
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]

async def initialize_components():
    """Initialize all application components"""
    global rag_pipeline, voice_processor, cache_manager, query_analyzer, document_processor
    
    logger.info("Initializing Inyandiko Legal AI Assistant...")
    
    try:
        # Initialize cache manager
        logger.info("Initializing cache manager...")
        cache_manager = EnterpriseCacheManager()
        await cache_manager.initialize()
        
        # Initialize document processor
        logger.info("Initializing document processor...")
        document_processor = DocumentIntelligenceProcessor()
        await document_processor.initialize()
        
        # Initialize query analyzer
        logger.info("Initializing query analyzer...")
        query_analyzer = ComprehensiveQueryAnalyzer()
        await query_analyzer.initialize()
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = AdvancedRAGPipeline(
            cache_manager=cache_manager,
            query_analyzer=query_analyzer,
            document_processor=document_processor
        )
        await rag_pipeline.initialize()
        
        # Initialize voice processor
        logger.info("Initializing voice processor...")
        voice_processor = SophisticatedVoiceProcessor(
            cache_manager=cache_manager
        )
        await voice_processor.initialize()
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

# Create FastAPI app
app = FastAPI(
    title="Inyandiko Legal AI Assistant",
    description="Voice-enabled Legal Information Assistant for Rwandan Law",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    await initialize_components()

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Shutting down application...")
    if cache_manager:
        await cache_manager.close()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components_status = {}
    
    # Check each component
    if rag_pipeline:
        components_status["rag_pipeline"] = "healthy" if await rag_pipeline.health_check() else "unhealthy"
    if voice_processor:
        components_status["voice_processor"] = "healthy" if await voice_processor.health_check() else "unhealthy"
    if cache_manager:
        components_status["cache_manager"] = "healthy" if await cache_manager.health_check() else "unhealthy"
    if query_analyzer:
        components_status["query_analyzer"] = "healthy" if await query_analyzer.health_check() else "unhealthy"
    if document_processor:
        components_status["document_processor"] = "healthy" if await document_processor.health_check() else "unhealthy"
    
    overall_status = "healthy" if all(status == "healthy" for status in components_status.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="2.0.0",
        components=components_status
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Inyandiko Legal AI Assistant API v2.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health"
    }

@app.post("/query", response_model=QueryResponse)
async def text_query(
    request: Request,
    query_request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """Process text-based legal query"""
    if not all([rag_pipeline, query_analyzer]):
        raise HTTPException(status_code=503, detail="Services not fully initialized")
    
    start_time = time.time()
    query_id = str(uuid.uuid4())
    session_id = query_request.session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Processing text query: {query_id}")
        
        # Check cache first
        if cache_manager:
            cached_response = await cache_manager.get_cached_response(
                query_request.query, 
                query_request.language
            )
            if cached_response:
                logger.info(f"Returning cached response for query: {query_id}")
                return QueryResponse(**cached_response)
        
        # Analyze query
        query_analysis = await query_analyzer.comprehensive_analyze(
            query=query_request.query,
            language=query_request.language,
            session_id=session_id
        )
        
        # Process through RAG pipeline
        rag_result = await rag_pipeline.enhanced_query(
            query=query_request.query,
            language=query_request.language,
            query_analysis=query_analysis,
            query_id=query_id
        )
        
        processing_time = time.time() - start_time
        
        # Format citations
        citations = []
        for doc in rag_result.get("source_documents", []):
            citations.append({
                "source_pdf": doc.metadata.get("source_pdf", "unknown"),
                "page_number": doc.metadata.get("page_number", 1),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        response = QueryResponse(
            answer=rag_result["answer"],
            citations=citations,
            confidence_score=rag_result.get("confidence_score", 0.8),
            processing_time=processing_time
        )
        
        # Cache response
        if cache_manager:
            await cache_manager.cache_response(
                query_request.query,
                query_request.language,
                response.dict()
            )
        
        logger.info(f"Query processed successfully: {query_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query {query_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")

@app.post("/voice_query")
async def voice_query(
    request: Request,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = None,
    language: str = "rw"
):
    """Process voice-based legal query"""
    if not all([rag_pipeline, voice_processor, query_analyzer]):
        raise HTTPException(status_code=503, detail="Services not fully initialized")
    
    start_time = time.time()
    query_id = str(uuid.uuid4())
    session_id = session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Processing voice query: {query_id}")
        
        if audio_file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="Audio file too large")
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Transcribe audio
        transcription_result = await voice_processor.enhanced_transcribe_audio(
            audio_data,
            expected_language=language,
            session_id=session_id
        )
        
        transcribed_text = transcription_result["text"]
        if not transcribed_text.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        logger.info(f"Audio transcribed: {transcribed_text[:100]}...")
        
        # Analyze query
        query_analysis = await query_analyzer.comprehensive_analyze(
            query=transcribed_text,
            language=language,
            session_id=session_id
        )
        
        # Process through RAG pipeline
        rag_result = await rag_pipeline.enhanced_query(
            query=transcribed_text,
            language=language,
            query_analysis=query_analysis,
            query_id=query_id
        )
        
        # Generate speech response
        async def generate_audio_response():
            async for audio_chunk in voice_processor.enhanced_text_to_speech(
                text=rag_result["answer"],
                language=language,
                session_id=session_id
            ):
                yield audio_chunk
        
        logger.info(f"Voice query processed successfully: {query_id}")
        return StreamingResponse(
            generate_audio_response(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=response.mp3"}
        )
        
    except Exception as e:
        logger.error(f"Error processing voice query {query_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing voice query")

@app.get("/get_pdf_page/{pdf_name}/{page_num}")
async def get_pdf_page(pdf_name: str, page_num: int):
    """Retrieve and render a specific page from a PDF document"""
    try:
        import fitz  # PyMuPDF
        
        # Security check - ensure path is within legal_docs directory
        legal_docs_dir = os.path.abspath("legal_docs")
        pdf_path = os.path.abspath(os.path.join(legal_docs_dir, pdf_name))
        
        if not pdf_path.startswith(legal_docs_dir):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF not found")
        
        # Open PDF and render page
        doc = fitz.open(pdf_path)
        
        if page_num < 1 or page_num > len(doc):
            raise HTTPException(status_code=404, detail="Page not found")
        
        page = doc[page_num - 1]  # Convert to 0-indexed
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("png")
        
        doc.close()
        
        return StreamingResponse(
            BytesIO(img_data),
            media_type="image/png"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving PDF page: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving PDF page")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=os.getenv("API_HOST", "0.0.0.0"), 
        port=int(os.getenv("API_PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
