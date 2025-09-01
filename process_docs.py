"""
process_docs.py: CLI script to bootstrap the knowledge base for Inyandiko Legal AI Assistant.

This script iterates through all documents in the source directory and uses the
DocumentIngestionService to process and index them into the SQLite database and
FAISS vector store.

This should be run once for initial setup, or to completely rebuild the knowledge base.
For live updates, the DirectoryWatcherService handles ingestion automatically.

Usage:
    # Bootstrap for the first time
    python process_docs.py

    # Force a complete rebuild, deleting all existing data
    python process_docs.py --force_rebuild
"""

import asyncio
import argparse
import logging
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import os

from document_processor import DocumentProcessor
from embedding_manager import AdvancedEmbeddingManager
from data_models import initialize_database, DB_PATH
from document_ingestion_service import DocumentIngestionService
from vector_store_manager import VectorStoreManager

# Configure logging for clear output during the script execution
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


async def clear_existing_knowledge_base():
    """
    Wipes the existing database and FAISS index for a clean rebuild.
    This is a destructive operation and should be used with caution.
    """
    logger.warning("--- CLEARING EXISTING KNOWLEDGE BASE ---")
    db_file = Path(DB_PATH)
    index_file = Path("vector_db/faiss_index.index")

    try:
        if db_file.exists():
            db_file.unlink()
            logger.info(f"Deleted database: {db_file}")
        if index_file.exists():
            index_file.unlink()
            logger.info(f"Deleted FAISS index: {index_file}")
    except OSError as e:
        logger.error(f"Error while deleting knowledge base files: {e}", exc_info=True)

    logger.warning("--- KNOWLEDGE BASE CLEARED ---")


async def main(args: argparse.Namespace):
    """Main function to bootstrap the knowledge base."""

    if args.force_rebuild:
        await clear_existing_knowledge_base()

    logger.info("--- Starting Knowledge Base Bootstrap Process ---")

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        logger.error(
            f"Documents directory not found: {docs_dir}. Please create it and add documents."
        )
        return

    # 1. Initialize database schema (creates the DB file and tables if they don't exist)
    await initialize_database()

    # 2. Initialize core components required for ingestion
    doc_processor = DocumentProcessor()
    await doc_processor.initialize()

    embedding_manager = AdvancedEmbeddingManager()
    await embedding_manager.initialize()  # Load the sentence-transformer model

    vector_store_manager = VectorStoreManager(
        index_path=Path("vector_db/faiss_index.index"),
        embedding_manager=embedding_manager,
    )
    await vector_store_manager.initialize()

    ingestion_service = DocumentIngestionService(
        doc_processor, embedding_manager, vector_store_manager
    )

    # 3. Find all supported documents in the source directory
    supported_extensions = {".pdf", ".docx", ".txt", ".md"}
    doc_files = [
        p
        for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]

    if not doc_files:
        logger.warning(
            f"No supported documents found in {docs_dir}. The knowledge base will be empty."
        )
        return

    logger.info(f"Found {len(doc_files)} documents to process.")

    # 4. Create and run ingestion tasks concurrently for efficiency
    tasks = [
        ingestion_service.process_and_index_document(str(doc_path)) for doc_path in doc_files
    ]

    # Use tqdm_asyncio for a real-time progress bar in the console
    await tqdm_asyncio.gather(*tasks, desc="Processing documents", unit="file")

    logger.info("--- Knowledge Base Bootstrap Process Completed Successfully ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap the Inyandiko knowledge base by processing and indexing all documents."
    )
    parser.add_argument(
        "--docs_dir",
        default="legal_docs",
        help="Directory containing the source legal documents.",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force a complete rebuild by deleting the existing database and vector index before starting.",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Bootstrap process interrupted by user.")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during the bootstrap process: {e}",
            exc_info=True,
        )
