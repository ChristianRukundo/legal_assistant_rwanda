"""
Service for ingesting documents into the system, including processing,
database updates, and vector indexing.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import List
from datetime import datetime

import aiosqlite
import structlog

from data_models import get_db_connection
from document_processor import DocumentProcessor, ProcessingStatus
from embedding_manager import AdvancedEmbeddingManager
from vector_store_manager import VectorStoreManager

logger = structlog.get_logger(__name__)


class DocumentIngestionService:
    """Handles the end-to-end process of ingesting and indexing documents."""

    def __init__(
        self,
        doc_processor: DocumentProcessor,
        embedding_manager: AdvancedEmbeddingManager,
        vector_store_manager: VectorStoreManager,
    ):
        self.doc_processor = doc_processor
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager

    async def process_and_index_document(self, file_path: str):
        """Processes a single document, updates the database, and indexes it."""
        logger.info(f"Starting ingestion for: {file_path}")
        file_path_obj = Path(file_path)

        try:
            file_hash = self._get_file_hash(file_path_obj)

            # The line below was `async with await get_db_connection() as db:`,
            # which caused a `RuntimeError: threads can only be started once`.
            # The `await` is incorrect because `aiosqlite.connect()` (which is
            # what get_db_connection wraps) returns a coroutine that also acts
            # as an async context manager. The `async with` statement handles
            # awaiting it correctly.
            async with get_db_connection() as db:
                db.row_factory = aiosqlite.Row

                cursor = await db.execute(
                    "SELECT id, file_hash FROM documents WHERE file_path = ?",
                    (str(file_path_obj),),
                )
                doc_record = await cursor.fetchone()

                if doc_record and doc_record["file_hash"] == file_hash:
                    logger.info(f"Document {file_path} is already up to date.")
                    return

                processed_docs = await self.doc_processor.batch_process([file_path])
                processed_doc = processed_docs[0]

                if processed_doc.metadata.status != ProcessingStatus.COMPLETED:
                    logger.error(
                        f"Failed to process {file_path}: {processed_doc.metadata.error_message}"
                    )
                    return

                if doc_record:
                    doc_id = doc_record["id"]
                    cursor = await db.execute(
                        "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
                    )
                    old_chunk_ids = [row["id"] for row in await cursor.fetchall()]
                    if old_chunk_ids:
                        await self.vector_store_manager.remove(old_chunk_ids)
                        await db.execute(
                            "DELETE FROM chunks WHERE doc_id = ?", (doc_id,)
                        )

                    await db.execute(
                        "UPDATE documents SET file_hash = ?, status = 'COMPLETED', updated_at = ? WHERE id = ?",
                        (file_hash, datetime.utcnow(), doc_id),
                    )
                else:
                    cursor = await db.execute(
                        "INSERT INTO documents (file_path, file_hash, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            str(file_path_obj),
                            file_hash,
                            "COMPLETED",
                            datetime.utcnow(),
                            datetime.utcnow(),
                        ),
                    )
                    doc_id = cursor.lastrowid

                if processed_doc.chunks:
                    chunk_data = [
                        (doc_id, chunk.content, chunk.page_number)
                        for chunk in processed_doc.chunks
                    ]
                    await db.executemany(
                        "INSERT INTO chunks (doc_id, content, page_number) VALUES (?, ?, ?)",
                        chunk_data,
                    )
                    await db.commit()

                    cursor = await db.execute(
                        "SELECT id, content FROM chunks WHERE doc_id = ?", (doc_id,)
                    )
                    new_chunks = await cursor.fetchall()
                    chunk_ids = [c["id"] for c in new_chunks]
                    chunk_contents = [c["content"] for c in new_chunks]
                    embeddings = self.embedding_manager.get_embeddings(chunk_contents)
                    await self.vector_store_manager.add(embeddings, chunk_ids)

                logger.info(f"Successfully ingested and indexed {file_path}")

        except Exception as e:
            logger.error(f"Error during ingestion of {file_path}: {e}", exc_info=True)
            raise

    def _get_file_hash(self, file_path: Path) -> str:
        """Computes the SHA256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
