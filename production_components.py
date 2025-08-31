"""
Production Components for Inyandiko Legal AI Assistant
Handles model orchestration and monitoring in production environment.
Version: 1.4 (Offline Gated Model Loading & Quantization)
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.schema import Document
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import os

logger = logging.getLogger(__name__)

class ProductionModelOrchestrator:
    """Orchestrates LLM model loading and response generation."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # --- FIX 1: Reverting back to the pre-downloaded Mistral model ---
        self.model_name = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        
        self.response_counter = Counter('model_responses_total', 'Total model responses', ['status'])
        self.response_duration = Histogram('model_response_duration_seconds', 'Response generation duration')
    
    async def initialize(self):
        """Loads the LLM model and tokenizer from the local cache."""
        logger.info(f"Initializing ProductionModelOrchestrator with model: {self.model_name} on {self.device}")
        try:
            loop = asyncio.get_running_loop()
            def load_model_sync():
                # Configuration for 4-bit quantization is essential for Mistral-7B on 8GB RAM
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
                    bnb_4bit_use_double_quant=False,
                )
                
                logger.info("Attempting to load model and tokenizer from local cache ONLY.")
                
                # --- FIX 2: Added local_files_only=True to prevent internet connection attempts ---
                # This forces transformers to use the files you've already downloaded.
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    local_files_only=True
                )
                return tokenizer, model
            
            self.tokenizer, self.model = await loop.run_in_executor(None, load_model_sync)
            logger.info("Model and tokenizer loaded successfully from local cache in 4-bit mode.")
            logger.info(f"Model memory footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")
        except EnvironmentError as e:
            logger.error(f"FATAL: Could not load model from local cache. The necessary files might be missing or corrupted. {e}")
            logger.error("Please ensure you have a stable internet connection and have successfully downloaded the model at least once.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the model: {e}", exc_info=True)
            raise
    
    async def generate_response(self, query: str, context_documents: List[Document], language: str, query_analysis: Dict) -> Dict[str, Any]:
        """Generates a response using the loaded LLM."""
        if not self.model or not self.tokenizer:
            logger.error("Cannot generate response: Model not initialized.")
            return {"answer": "I am sorry, but the language model is not available at the moment. Please try again later."}
        
        start_time = datetime.now()
        try:
            context = "\n\n".join([doc.page_content for doc in context_documents])
            max_context_length = 3000
            if len(self.tokenizer.encode(context)) > max_context_length:
                tokenized_context = self.tokenizer.encode(context)
                truncated_tokens = tokenized_context[:max_context_length]
                context = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True) + "..."

            # --- FIX 3: Reverting to the correct prompt format for Mistral Instruct models ---
            prompt = f"""<s>[INST] You are a helpful legal AI assistant specializing in Rwandan law. 
Your task is to provide a clear and accurate response in {language} based *only* on the provided context. Do not use any prior knowledge. If the context does not contain the answer, state that you cannot find the information in the provided documents.

Context:
---
{context}
---

Question: {query} [/INST]"""

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            generation_args = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            }
            
            outputs = self.model.generate(**inputs, **generation_args)
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated response, not the prompt
            answer = full_text.split("[/INST]")[-1].strip()
            
            duration = (datetime.now() - start_time).total_seconds()
            self.response_counter.labels(status='success').inc()
            self.response_duration.observe(duration)
            
            return {"answer": answer}
        
        except Exception as e:
            self.response_counter.labels(status='error').inc()
            logger.error(f"Failed to generate response: {e}", exc_info=True)
            return {"answer": "I am sorry, but I encountered an error while generating a response."}
    
    async def health_check(self) -> bool:
        """Checks if the model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None

class ProductionMonitoringEngine:
    """Handles monitoring and metrics logging using Prometheus."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.query_counter = Counter('queries_total', 'Total queries processed', ['status'])
        self.query_duration = Histogram('query_duration_seconds', 'Query processing duration')
        self.confidence_gauge = Gauge('query_confidence_score', 'Last query confidence score')
        self.server_started = False
    
    async def initialize(self):
        """Starts the Prometheus HTTP server."""
        if not self.server_started:
            try:
                start_http_server(self.port)
                self.server_started = True
                logger.info(f"Prometheus metrics server started on port {self.port}")
            except OSError as e:
                logger.warning(f"Could not start Prometheus server on port {self.port} (maybe it's already running?): {e}")
                self.server_started = True

    async def log_query_metrics(self, metrics: Dict[str, Any]):
        """Logs query metrics to Prometheus."""
        status = 'success' if metrics.get('status', 'success') == 'success' else 'error'
        self.query_counter.labels(status=status).inc()
        if 'processing_time' in metrics:
            self.query_duration.observe(metrics['processing_time'])
        if 'confidence_score' in metrics:
            self.confidence_gauge.set(metrics['confidence_score'])
        
        logger.debug(f"Logged query metrics: {metrics}")
    
    async def health_check(self) -> bool:
        """Checks if monitoring is active."""
        return self.server_started