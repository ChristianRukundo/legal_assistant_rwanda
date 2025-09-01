"""
VectorStoreManager for handling the FAISS index with support for additions and removals.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss

from embedding_manager import AdvancedEmbeddingManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the FAISS vector index for efficient updates."""

    def __init__(self, index_path: Path, embedding_manager: AdvancedEmbeddingManager):
        self.index_path = Path(index_path)
        self.embedding_manager = embedding_manager
        self.index: faiss.IndexIDMap | None = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Loads the index from disk or creates a new one."""
        async with self._lock:
            if self._initialized:
                return
            try:
                # Get embedding dimension dynamically
                sample_embedding = self.embedding_manager.get_embeddings(["test"])
                self.dimension = sample_embedding.shape[1]

                if self.index_path.exists():
                    logger.info(f"Loading existing FAISS index from {self.index_path}")
                    self.index = faiss.read_index(str(self.index_path))
                else:
                    logger.info(
                        f"No FAISS index found. Creating a new one with dimension {self.dimension}."
                    )
                    # Use IndexIDMap to map our database chunk IDs to vectors
                    cpu_index = faiss.IndexFlatL2(self.dimension)
                    self.index = faiss.IndexIDMap(cpu_index)

                self._initialized = True
                if self.index is not None:
                    logger.info(
                        f"Vector store initialized. Index contains {self.index.ntotal} vectors."
                    )
                else:
                    logger.warning("Vector store initialized, but index is None.")

            except Exception as e:
                logger.error(
                    f"Failed to initialize VectorStoreManager: {e}", exc_info=True
                )
                raise

    async def add(self, vectors: np.ndarray, ids: List[int]):
        """Adds vectors with their corresponding database IDs to the index."""
        if not self._initialized or self.index is None:
            raise RuntimeError("VectorStoreManager not initialized.")

        async with self._lock:
            vector_ids = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(vectors.astype(np.float32), xids=vector_ids)  # type: ignore
            await self._save_index()
            logger.info(
                f"Added {len(vectors)} vectors. Index now has {self.index.ntotal} total vectors."
            )

    async def remove(self, ids: List[int]):
        """Removes vectors by their database IDs from the index."""
        if not self._initialized or self.index is None:
            raise RuntimeError("VectorStoreManager not initialized.")

        async with self._lock:
            if not ids:
                return
            ids_to_remove = np.array(ids, dtype=np.int64)
            removed_count = self.index.remove_ids(ids_to_remove)
            if removed_count > 0:
                await self._save_index()
            logger.info(
                f"Removed {removed_count} vectors. Index now has {self.index.ntotal} total vectors."
            )

    async def search(
        self, query_vector: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Searches the index for the k nearest neighbors."""
        if not self._initialized or self.index is None or self.index.ntotal == 0:
            return np.array([]), np.array([])

        distances, ids = self.index.search(query_vector.astype(np.float32), k)  # type: ignore
        return distances, ids

    async def _save_index(self):
        """Saves the current state of the index to disk."""
        if self.index:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            logger.debug(f"FAISS index saved to {self.index_path}")
