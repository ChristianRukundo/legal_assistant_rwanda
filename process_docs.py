"""
process_docs.py: Script to process legal documents and build the vector database for Inyandiko Legal AI Assistant.

This script:
- Loads documents from a specified directory (default: legal_docs/).
- Processes them using AdvancedDocumentProcessor (handles PDFs, DOCX, etc., with OCR, chunking, metadata extraction).
- Embeds chunks using Sentence Transformers.
- Builds and saves FAISS index along with document metadata.
- Supports async processing for efficiency.
- Includes progress tracking, error handling, and logging.
- CLI args for customization.

Usage:
    python process_docs.py --docs_dir legal_docs --vector_db_dir vector_db --log_level INFO

Requirements:
- Run in the project root.
- .env file with configs (e.g., EMBEDDING_MODEL=intfloat/multilingual-e5-large).
- legal_docs/ must contain files.

Production Notes:
- Idempotent: Skips if index exists unless --force_rebuild.
- Handles large dirs: Batch processing to avoid OOM.
- Graceful shutdown: Catches KeyboardInterrupt.
- Metrics: Logs processing time, doc count, etc.
"""

import asyncio
import argparse
import logging
import os
import json
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import structlog
from tqdm.asyncio import tqdm_asyncio  # For async progress bars
import faiss
import numpy as np

from advanced_rag_pipeline import AdvancedRAGPipeline
from advanced_document_processor import AdvancedDocumentProcessor, ProcessingStatus
from enterprise_caching_system import CacheManager
from intelligent_query_processor import IntelligentQueryProcessor
from production_components import ProductionModelOrchestrator, ProductionMonitoringEngine

# Load environment variables
load_dotenv()

# Configure logging with structlog
logger = structlog.get_logger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

async def main(args: argparse.Namespace) -> None:
    start_time = time.monotonic()
    logger.info("Starting document processing script...")

    # Validate paths
    docs_dir = Path(args.docs_dir).resolve()
    vector_db_dir = Path(args.vector_db_dir).resolve()
    if not docs_dir.exists() or not docs_dir.is_dir():
        logger.error(f"Documents directory does not exist: {docs_dir}")
        return
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    # Check if index already exists
    index_path = vector_db_dir / "faiss_index.index"
    docs_json_path = vector_db_dir / "documents.json"
    if index_path.exists() and docs_json_path.exists() and not args.force_rebuild:
        logger.info(f"Vector DB already exists at {vector_db_dir}. Skipping rebuild. Use --force_rebuild to override.")
        return

    # Initialize components
    cache_manager = CacheManager()  # Optional: For caching embeddings if heavy
    await cache_manager.initialize()

    document_processor = AdvancedDocumentProcessor()

    query_analyzer = IntelligentQueryProcessor()
    await query_analyzer.initialize()

    model_orchestrator = ProductionModelOrchestrator()
    await model_orchestrator.initialize()

    monitoring_engine = ProductionMonitoringEngine()
    await monitoring_engine.initialize()

    rag_pipeline = AdvancedRAGPipeline(
        cache_manager=cache_manager,
        query_analyzer=query_analyzer,
        document_processor=document_processor,
        model_orchestrator=model_orchestrator,
        monitoring_engine=monitoring_engine
    )
    await rag_pipeline.initialize()

    # Gather document files
    supported_extensions = {'.pdf', '.docx', '.txt', '.html', '.md', '.xlsx', '.csv', '.json', '.xml', '.epub', '.zip', '.rar', '.7z'}
    doc_files: List[Path] = [f for f in docs_dir.rglob('*') if f.is_file() and f.suffix.lower() in supported_extensions]
    if not doc_files:
        logger.warning(f"No supported documents found in {docs_dir}. Exiting.")
        return

    logger.info(f"Found {len(doc_files)} documents to process.")

    # Process documents in batches to avoid OOM
    batch_size = int(os.getenv("BATCH_SIZE", 32))
    processed_docs = []
    failed_docs = []

    async def process_batch(batch: List[Path]) -> List:
        tasks = [rag_pipeline.document_processor.process_document(str(file)) for file in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)

    for i in range(0, len(doc_files), batch_size):
        batch = doc_files[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(doc_files) + batch_size - 1)//batch_size} ({len(batch)} docs)...")
        
        results = await tqdm_asyncio.gather(process_batch(batch), desc="Processing batch")
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing document: {result}")
                failed_docs.append(str(result))
            else:
                if result.metadata.status == ProcessingStatus.COMPLETED:
                    processed_docs.append(result)
                else:
                    logger.warning(f"Document processing failed: {result.metadata.file_name} - {result.metadata.error_message}")
                    failed_docs.append(result.metadata.file_name)

    if not processed_docs:
        logger.error("No documents processed successfully. Exiting.")
        return

    # Embed and build index
    logger.info(f"Embedding {len(processed_docs)} processed documents...")
    all_chunks = [chunk for doc in processed_docs for chunk in doc.chunks]
    embeddings = rag_pipeline.embedding_manager.get_embeddings([chunk.content for chunk in all_chunks])

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, str(index_path))
    with open(docs_json_path, 'w', encoding='utf-8') as f:
        json.dump([doc.__dict__ for doc in processed_docs], f, default=str, indent=2)  # Serialize dataclass

    processing_time = time.monotonic() - start_time
    logger.info(f"Vector DB built successfully at {vector_db_dir}. Processed {len(processed_docs)} docs in {processing_time:.2f}s.")
    if failed_docs:
        logger.warning(f"{len(failed_docs)} documents failed: {failed_docs}")

    # Cleanup
    await cache_manager.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process legal documents and build vector DB.")
    parser.add_argument("--docs_dir", default="legal_docs", help="Directory containing legal documents.")
    parser.add_argument("--vector_db_dir", default="vector_db", help="Directory to save vector DB.")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild even if index exists.")
    parser.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, etc.).")

    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Cleaning up...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)   