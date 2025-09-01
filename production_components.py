"""
Production Components for Inyandiko Legal AI Assistant
Handles model orchestration (now with LlamaCpp for CPU) and monitoring.
Version: 1.8 (CPU / GGUF Optimized)
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_community.llms import LlamaCpp
from langchain.schema import Document
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import os
import structlog

logger = structlog.get_logger(__name__)


class ProductionModelOrchestrator:
    """Orchestrates LLM model loading and response generation using LlamaCpp."""

    def __init__(self):
        self.model: Optional[LlamaCpp] = None
        
        
        self.model_path = os.getenv(
            "LLM_MODEL_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        )

        self.response_counter = Counter(
            "model_responses_total", "Total model responses", ["status", "type"]
        )
        self.response_duration = Histogram(
            "model_response_duration_seconds", "Response generation duration"
        )

    async def initialize(self):
        """Loads the GGUF model using LlamaCpp in a memory-efficient way."""
        logger.info(
            "Initializing ProductionModelOrchestrator with GGUF model",
            model_path=self.model_path,
        )

        if not os.path.exists(self.model_path):
            logger.error("FATAL: GGUF model file not found.", path=self.model_path)
            logger.error(
                "Please download the 'mistral-7b-instruct-v0.2.Q4_K_M.gguf' model from TheBloke on Hugging Face and place it in the 'models' directory."
            )
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            loop = asyncio.get_running_loop()

            
            llm_params = {
                "model_path": self.model_path,
                "n_ctx": 4096,  
                "n_gpu_layers": 0,  
                "n_batch": 512,  
                "temperature": 0.7,  
                "max_tokens": 1024,  
                "top_p": 0.9,
                "verbose": False,  
            }

            
            
            def load_model_sync() -> LlamaCpp:
                return LlamaCpp(**llm_params)

            self.model = await loop.run_in_executor(None, load_model_sync)
            logger.info("GGUF model loaded successfully via LlamaCpp.")

        except Exception as e:
            logger.error(
                "An unexpected error occurred while loading the GGUF model",
                error=str(e),
                exc_info=True,
            )
            raise

    async def _generate(self, prompt: str) -> str:
        """Internal helper to run the blocking generation task in a thread pool."""
        if not self.model:
            logger.error("Model generation called before initialization.")
            raise RuntimeError("Model is not initialized.")

        loop = asyncio.get_running_loop()
        
        
        return await loop.run_in_executor(None, self.model.invoke, prompt)

    async def generate_raw(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generates a raw text response. Note: max_new_tokens is set during LlamaCpp init."""
        return await self._generate(prompt)

    async def generate_response(
        self,
        query: str,
        context_documents: List[Document],
        language: str,
        conversation_summary: str,
    ) -> Dict[str, Any]:
        """Generates a final, user-facing response using the loaded GGUF model."""
        start_time = datetime.now()
        try:
            context_str = (
                "\n\n".join([doc.page_content for doc in context_documents])
                if context_documents
                else "No context documents found."
            )

            
            prompt = f"""<s>[INST] You are a helpful legal AI assistant specializing in Rwandan law. 
Your task is to provide a clear and accurate response in {language} based *only* on the provided context. Do not use any prior knowledge. If the context does not contain the answer, state that you cannot find the information in the provided documents.

CONVERSATION SUMMARY:
---
{conversation_summary}
---

CONTEXT DOCUMENTS:
---
{context_str}
---

Question: {query} [/INST]"""

            answer = await self._generate(prompt)

            duration = (datetime.now() - start_time).total_seconds()
            self.response_counter.labels(status="success", type="rag").inc()
            self.response_duration.observe(duration)

            return {"answer": answer.strip()}

        except Exception as e:
            self.response_counter.labels(status="error", type="rag").inc()
            logger.error(
                "Failed to generate RAG response with LlamaCpp",
                error=str(e),
                exc_info=True,
            )
            return {
                "answer": "I am sorry, but I encountered an error while generating a response."
            }

    async def health_check(self) -> bool:
        """Checks if the model is loaded and ready."""
        return self.model is not None


class ProductionMonitoringEngine:
    """Handles monitoring and metrics logging using Prometheus."""

    def __init__(self, port: int = 8001):
        self.port = port
        self.query_counter = Counter(
            "queries_total", "Total queries processed", ["status"]
        )
        self.query_duration = Histogram(
            "query_duration_seconds", "Query processing duration"
        )
        self.confidence_gauge = Gauge(
            "query_confidence_score", "Last query confidence score"
        )
        self.server_started = False
        self.event_counter = Counter(
            "system_events_total", "Total discrete system events", ["event_name"]
        )

    async def initialize(self):
        """Starts the Prometheus HTTP server."""
        if not self.server_started:
            try:
                start_http_server(self.port)
                self.server_started = True
                logger.info(f"Prometheus metrics server started on port {self.port}")
            except OSError as e:
                logger.warning(
                    f"Could not start Prometheus server on port {self.port} (maybe it's already running?): {e}"
                )
                self.server_started = True

    async def log_event(self, event_name: str, details: Dict[str, Any] = {}):
        """
        Logs a discrete event for monitoring and operational tracking.

        Args:
            event_name: The name of the event (e.g., 'self_correction_triggered').
            details: An optional dictionary of details to include in structured logs.
        """
        self.event_counter.labels(event_name=event_name).inc()
        log_details = details if details else {}
        logger.info(f"Event logged: {event_name}", **log_details)

    async def log_query_metrics(self, metrics: Dict[str, Any]):
        """Logs query metrics to Prometheus."""
        status = "success" if metrics.get("status", "success") == "success" else "error"
        self.query_counter.labels(status=status).inc()
        if "processing_time" in metrics:
            self.query_duration.observe(metrics["processing_time"])
        if "confidence_score" in metrics:
            self.confidence_gauge.set(metrics["confidence_score"])

        logger.debug(f"Logged query metrics: {metrics}")

    async def health_check(self) -> bool:
        """Checks if monitoring is active."""
        return self.server_started
