import os
import asyncio
import uuid
import time
import hashlib
from typing import List, Optional, Dict, Any, AsyncGenerator, cast
import logging
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# --- FIX 2: More explicit import to help Pylance ---
import fitz
from fitz.fitz import Document as FitzDocument

from advanced_rag_pipeline import AdvancedRAGPipeline
from sophisticated_voice_processor import SophisticatedVoiceProcessor
from enterprise_caching_system import CacheManager
from intelligent_query_processor import IntelligentQueryProcessor, ProcessedQuery
from advanced_document_processor import AdvancedDocumentProcessor
from production_components import ProductionModelOrchestrator, ProductionMonitoringEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Centralized application state
app_state: Dict[str, Any] = {}

async def initialize_components() -> None:
    """Initialize all application components and store them in the app_state dictionary."""
    logger.info("Initializing Inyandiko Legal AI Assistant...")
    try:
        cache_manager = CacheManager()
        await cache_manager.initialize()
        app_state["cache_manager"] = cache_manager
        
        document_processor = AdvancedDocumentProcessor()
        app_state["document_processor"] = document_processor
        
        query_analyzer = IntelligentQueryProcessor()
        await query_analyzer.initialize()
        app_state["query_analyzer"] = query_analyzer
        
        model_orchestrator = ProductionModelOrchestrator()
        await model_orchestrator.initialize()
        app_state["model_orchestrator"] = model_orchestrator
        
        monitoring_engine = ProductionMonitoringEngine()
        await monitoring_engine.initialize()
        app_state["monitoring_engine"] = monitoring_engine
        
        rag_pipeline = AdvancedRAGPipeline(
            cache_manager=app_state["cache_manager"],
            query_analyzer=app_state["query_analyzer"],
            document_processor=app_state["document_processor"],
            model_orchestrator=app_state["model_orchestrator"],
            monitoring_engine=app_state["monitoring_engine"]
        )
        await rag_pipeline.initialize()
        app_state["rag_pipeline"] = rag_pipeline
        
        voice_processor = SophisticatedVoiceProcessor(
            cache_manager=app_state["cache_manager"],
            model_orchestrator=app_state["model_orchestrator"],
            monitoring_engine=app_state["monitoring_engine"]
        )
        await voice_processor.initialize()
        app_state["voice_processor"] = voice_processor
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize components during startup: {e}", exc_info=True)
        raise

async def close_components() -> None:
    """Gracefully close components during shutdown."""
    logger.info("Shutting down application components...")
    if "cache_manager" in app_state and app_state["cache_manager"]:
        await app_state["cache_manager"].close()
    logger.info("Shutdown complete.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_components()
    yield
    await close_components()

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The legal query text.")
    language: str = Field(default="rw", pattern="^(rw|en|fr)$", description="Language of the query (rw, en, fr).")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation context.")

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

app = FastAPI(
    title="Inyandiko Legal AI Assistant",
    description="Voice-enabled Legal Information Assistant for Rwandan Law",
    version="2.1.1-type-safe",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Performs a health check on all critical system components."""
    components_status: Dict[str, str] = {}
    component_names = [
        "rag_pipeline", "voice_processor", "cache_manager", "query_analyzer",
        "document_processor", "model_orchestrator", "monitoring_engine"
    ]
    
    for name in component_names:
        component = app_state.get(name)
        if not component:
            components_status[name] = "uninitialized"
            continue
        
        if hasattr(component, 'health_check'):
            is_healthy = await component.health_check() if asyncio.iscoroutinefunction(component.health_check) else component.health_check()
            components_status[name] = "healthy" if is_healthy else "unhealthy"
        else:
            components_status[name] = "healthy"
    
    overall_status = "healthy" if all(s == "healthy" for s in components_status.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=app.version,
        components=components_status
    )

@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "message": f"Inyandiko Legal AI Assistant API v{app.version}",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health"
    }

@app.post("/query", response_model=QueryResponse)
async def text_query(query_request: QueryRequest) -> QueryResponse:
    """Processes a text-based legal query."""
    components = {
        "rag_pipeline": app_state.get("rag_pipeline"),
        "query_analyzer": app_state.get("query_analyzer"),
        "cache_manager": app_state.get("cache_manager")
    }
    if not all(components.values()):
        raise HTTPException(status_code=503, detail="Services are not fully initialized. Please try again shortly.")

    # --- FIX 1: Use typing.cast to inform the linter the types are now guaranteed ---
    rag_pipeline = cast(AdvancedRAGPipeline, components["rag_pipeline"])
    query_analyzer = cast(IntelligentQueryProcessor, components["query_analyzer"])
    cache_manager = cast(CacheManager, components["cache_manager"])

    start_time = time.time()
    query_id = str(uuid.uuid4())
    session_id = query_request.session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Processing text query: {query_id}")
        
        cache_key = f"query:{hashlib.md5(f'{query_request.query}:{query_request.language}'.encode()).hexdigest()}"
        cached_response = await cache_manager.cache.get(cache_key)
        if cached_response and isinstance(cached_response, dict):
            logger.info(f"Returning cached response for query: {query_id}")
            return QueryResponse(**cached_response)
        
        query_analysis = await query_analyzer.comprehensive_analyze(
            query=query_request.query, language=query_request.language, session_id=session_id
        )
        
        rag_result = await rag_pipeline.enhanced_query(
            query=query_request.query, language=query_request.language, query_analysis=query_analysis, query_id=query_id
        )
        
        processing_time = time.time() - start_time
        
        citations = [{
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page_number": doc.metadata.get("page_number", 1),
            "excerpt": doc.page_content[:250] + "..."
        } for doc in rag_result.get("source_documents", [])]
        
        response = QueryResponse(
            answer=rag_result["answer"],
            citations=citations,
            confidence_score=rag_result.get("confidence_score", 0.0),
            processing_time=processing_time
        )
        
        await cache_manager.cache.set(cache_key, response.model_dump(), ttl=3600)
        
        logger.info(f"Query processed successfully: {query_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query {query_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.")

@app.post("/voice_query")
async def voice_query(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Query(None),
    language: str = Query("rw", pattern="^(rw|en|fr)$")
) -> StreamingResponse:
    """Processes a voice-based legal query and streams back an audio response."""
    components = {
        "rag_pipeline": app_state.get("rag_pipeline"),
        "voice_processor": app_state.get("voice_processor"),
        "query_analyzer": app_state.get("query_analyzer")
    }
    if not all(components.values()):
        raise HTTPException(status_code=503, detail="Services not fully initialized.")

    rag_pipeline = cast(AdvancedRAGPipeline, components["rag_pipeline"])
    voice_processor = cast(SophisticatedVoiceProcessor, components["voice_processor"])
    query_analyzer = cast(IntelligentQueryProcessor, components["query_analyzer"])
    
    query_id = str(uuid.uuid4())
    session_id = session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Processing voice query: {query_id}")
        
        if audio_file.size and audio_file.size > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio file is too large. Limit is 15MB.")
        
        audio_data = await audio_file.read()
        
        transcription_result = await voice_processor.enhanced_transcribe_audio(
            audio_data=audio_data, expected_language=language
        )
        
        transcribed_text = transcription_result["text"]
        if not transcribed_text or not transcribed_text.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio. Please speak clearly and try again.")
        
        logger.info(f"Audio transcribed for query {query_id}: {transcribed_text[:100]}...")
        
        query_analysis = await query_analyzer.comprehensive_analyze(
            query=transcribed_text, language=language, session_id=session_id
        )
        
        rag_result = await rag_pipeline.enhanced_query(
            query=transcribed_text, language=language, query_analysis=query_analysis, query_id=query_id
        )
        
        async def generate_audio_stream() -> AsyncGenerator[bytes, None]:
            audio_bytes = await voice_processor.enhanced_text_to_speech(
                text=rag_result["answer"], language=language
            )
            yield audio_bytes

        logger.info(f"Voice query processed successfully: {query_id}")
        return StreamingResponse(generate_audio_stream(), media_type="audio/mpeg")
        
    except Exception as e:
        logger.error(f"Error processing voice query {query_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the voice query.")

@app.get("/get_pdf_page/{pdf_name}/{page_num}")
async def get_pdf_page(pdf_name: str, page_num: int) -> StreamingResponse:
    """Retrieves and renders a specific page from a PDF document in the knowledge base."""
    try:
        legal_docs_dir = Path("legal_docs").resolve()
        pdf_path = (legal_docs_dir / pdf_name).resolve()
        
        if not pdf_path.is_relative_to(legal_docs_dir):
            raise HTTPException(status_code=403, detail="Access to the requested resource is forbidden.")
        
        if not pdf_path.is_file():
            raise HTTPException(status_code=404, detail=f"PDF document '{pdf_name}' not found.")
        
        doc: FitzDocument = fitz.Document(pdf_path)
        
        if not (1 <= page_num <= len(doc)):
            doc.close()
            raise HTTPException(status_code=404, detail=f"Page number {page_num} is out of bounds for this document.")
        
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        doc.close()
        
        return StreamingResponse(BytesIO(img_data), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error retrieving PDF page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while retrieving the PDF page.")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"), 
        port=int(os.getenv("API_PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=True
    )