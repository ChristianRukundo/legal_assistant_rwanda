"""
DocumentIngestionService: Handles the end-to-end process of ingesting a single document
into the knowledge base, including processing, database updates, and vector indexing.
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import List

from document_processor import DocumentProcessor, ProcessedDocument, ProcessingStatus
from data_models import get_db_connection
from vector_store_manager import VectorStoreManager
from rag_pipeline import AdvancedEmbeddingManager

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    def __init__(
        self,
        doc_processor: DocumentProcessor,
        embedding_manager: AdvancedEmbeddingManager,
        vector_store_manager: VectorStoreManager,
    ):
        self.doc_processor = doc_processor
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculates the SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def process_and_index_document(self, file_path: Path):
        """Processes a single document and updates the knowledge base."""
        logger.info(f"Starting ingestion for: {file_path}")
        if not file_path.exists():
            logger.warning(
                f"File not found: {file_path}, attempting to delete from knowledge base."
            )
            await self.delete_document(str(file_path))
            return

        try:
            file_hash = self._calculate_file_hash(file_path)
            async with await get_db_connection() as db:

                existing_doc = await db.execute(
                    "SELECT id, file_hash FROM documents WHERE file_path = ?",
                    (str(file_path),),
                )
                doc_row = await existing_doc.fetchone()

                if doc_row and doc_row["file_hash"] == file_hash:
                    logger.info(
                        f"Document {file_path} is already up to date. Skipping."
                    )
                    return

                if doc_row:
                    logger.info(
                        f"Document {file_path} has been modified. Re-indexing..."
                    )
                    await self._delete_document_data(db, doc_row["id"])

                processed_doc = await self.doc_processor.process_document(
                    str(file_path)
                )

                async with db.executescript("BEGIN TRANSACTION;"):
                    if processed_doc.metadata.status == ProcessingStatus.COMPLETED:

                        cursor = await db.execute(
                            "INSERT INTO documents (file_path, file_hash, status) VALUES (?, ?, ?)",
                            (str(file_path), file_hash, "COMPLETED"),
                        )
                        doc_id = cursor.lastrowid

                        chunk_data = [
                            (
                                doc_id,
                                chunk.content,
                                chunk.page_number,
                                chunk.chunk_index,
                            )
                            for chunk in processed_doc.chunks
                        ]
                        if chunk_data:
                            await db.executemany(
                                "INSERT INTO chunks (doc_id, content, page_number, chunk_index) VALUES (?, ?, ?, ?)",
                                chunk_data,
                            )

                            chunk_rows = await db.execute(
                                "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
                            )
                            chunk_ids = [
                                row["id"] for row in await chunk_rows.fetchall()
                            ]

                            chunk_contents = [
                                chunk.content for chunk in processed_doc.chunks
                            ]
                            embeddings = self.embedding_manager.get_embeddings(
                                chunk_contents
                            )
                            await self.vector_store_manager.add(embeddings, chunk_ids)
                    else:
                        logger.warning(
                            f"Processing failed for {file_path}. Status: {processed_doc.metadata.status.value}"
                        )
                        await db.execute(
                            "INSERT INTO documents (file_path, file_hash, status) VALUES (?, ?, ?)",
                            (
                                str(file_path),
                                file_hash,
                                processed_doc.metadata.status.value,
                            ),
                        )

            logger.info(f"Successfully ingested and indexed document: {file_path}")

        except Exception as e:
            logger.error(f"Error during ingestion of {file_path}: {e}", exc_info=True)

    async def delete_document(self, file_path: str):
        """Deletes a document and all its associated data from the knowledge base."""
        logger.info(f"Deleting document from knowledge base: {file_path}")
        async with await get_db_connection() as db:
            async with db.executescript("BEGIN TRANSACTION;"):
                cursor = await db.execute(
                    "SELECT id FROM documents WHERE file_path = ?", (file_path,)
                )
                doc_row = await cursor.fetchone()
                if doc_row:
                    await self._delete_document_data(db, doc_row["id"])
                    logger.info(f"Successfully deleted document: {file_path}")
                else:
                    logger.warning(
                        f"Document not found in knowledge base for deletion: {file_path}"
                    )

    async def _delete_document_data(self, db, doc_id: int):
        """Internal helper to delete a document's chunks, vectors, and metadata."""

        cursor = await db.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
        chunk_ids_to_remove = [row["id"] for row in await cursor.fetchall()]

        if chunk_ids_to_remove:
            await self.vector_store_manager.remove(chunk_ids_to_remove)

        await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
