
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import structlog

logger = structlog.get_logger(__name__)

class AdvancedEmbeddingManager:
    """Manages multiple embedding models for different use cases."""

    def __init__(self):
        self.models: Dict[str, Optional[Union[SentenceTransformer, CrossEncoder]]] = {}

        self.device_string = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_configs = {
            "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "semantic": "sentence-transformers/all-MiniLM-L6-v2",
        }

    async def initialize(self):
        """Initializes all embedding models asynchronously."""
        logger.info("Initializing advanced embedding models...")
        tasks = [
            self._load_model(name, path) for name, path in self.model_configs.items()
        ]
        await asyncio.gather(*tasks)
        logger.info("All embedding models load attempts completed.")

    async def _load_model(self, model_name: str, model_path: str):
        """Loads a single model into memory, running blocking operations in an executor."""
        try:
            loop = asyncio.get_running_loop()

            def load_sync():
                if "cross_encoder" in model_name:
                    return CrossEncoder(model_path, device=self.device_string)
                else:
                    return SentenceTransformer(model_path, device=self.device_string)

            self.models[model_name] = await loop.run_in_executor(None, load_sync)
            logger.info(
                f"Loaded model '{model_name}' on device '{self.device_string}'."
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
            self.models[model_name] = None

    def get_embeddings(
        self, texts: List[str], model_name: str = "semantic"
    ) -> np.ndarray:
        """Generates embeddings using a specified model, with fallback for failures."""
        model = self.models.get(model_name)
        if not isinstance(model, SentenceTransformer) or model is None:
            logger.error(
                f"Embedding model '{model_name}' not available or is not a SentenceTransformer. Returning zero embeddings."
            )

            return np.zeros((len(texts), 384), dtype=np.float32)

        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings

    def rerank_documents(
        self, query: str, documents_content: List[str], top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Reranks documents using a CrossEncoder model for higher precision.
        `documents_content` is a list of strings (document texts).
        """
        model = self.models.get("cross_encoder")
        if not isinstance(model, CrossEncoder) or model is None:
            logger.warning(
                "Cross-encoder model not available for reranking. Returning unreranked documents (initial order with dummy scores)."
            )
            return [(i, 1.0) for i in range(min(top_k, len(documents_content)))]

        pairs = [[query, doc_content] for doc_content in documents_content]
        scores = model.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1]

        return [(int(idx), float(scores[idx])) for idx in ranked_indices[:top_k]]
