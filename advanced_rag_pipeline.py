# advanced_rag_pipeline.py
from __future__ import annotations # Enable postponed evaluation of type annotations

import asyncio
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import os

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.schema import Document

from langchain_community.retrievers import BM25Retriever
import spacy
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import structlog

# Import actual production components (received via dependency injection)
from enterprise_caching_system import CacheManager
from intelligent_query_processor import IntelligentQueryProcessor, ProcessedQuery, QueryComplexity, EmotionalTone # Specific imports for types
from advanced_document_processor import AdvancedDocumentProcessor, ProcessedDocument, ProcessingStatus
from production_components import ProductionModelOrchestrator, ProductionMonitoringEngine

# Configure structlog for consistent logging across the system
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = structlog.get_logger(__name__)

@dataclass
class RetrievalConfig:
    """Centralized configuration for advanced retrieval strategies."""
    max_initial_retrieval_docs: int = 50
    max_reranked_docs: int = 10
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    min_overlap_for_cross_ref: float = 0.1  # Threshold for cosine similarity for cross-referencing
    semantic_clustering_eps: float = 0.3
    semantic_clustering_min_samples: int = 2
    enable_cross_encoder_reranking: bool = True
    enable_multi_query_expansion: bool = True
    enable_contextual_compression: bool = False
    enable_document_graph_analysis: bool = True

@dataclass
class QueryContext:
    """Internal RAG query context, derived from IntelligentQueryProcessor's output."""
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    intent_classification: Dict[str, float] = field(default_factory=dict)
    entity_extraction: List[Dict[str, str]] = field(default_factory=list)
    legal_domain: str = ""
    complexity_score: float = 0.0 # Numeric representation of complexity
    urgency_level: str = "normal" # String representation of emotional tone/urgency
    user_expertise: str = "general" # Can be derived or passed from session
    session_context: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    """Comprehensive retrieval result with detailed metadata."""
    documents: List[Document]
    relevance_scores: List[float]
    confidence_score: float
    retrieval_strategy: str
    processing_time: float
    query_analysis: Dict[str, Any] = field(default_factory=dict) # The original ProcessedQuery as dict
    document_clusters: List[List[int]] = field(default_factory=list)
    cross_references: Dict[int, List[int]] = field(default_factory=dict)
    legal_citations: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedEmbeddingManager:
    """Manages multiple embedding models for different use cases within the RAG pipeline."""
    def __init__(self):
        self.models: Dict[str, Optional[Union[SentenceTransformer, CrossEncoder]]] = {}
        self.device_string: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_configs: Dict[str, str] = {
            "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "semantic": "sentence-transformers/all-MiniLM-L6-v2"
        }
        self._parent_rag_pipeline: Optional[AdvancedRAGPipeline] = None # For graph builder reference

    async def initialize(self):
        """Initializes all embedding models asynchronously."""
        logger.info("Initializing advanced embedding models...")
        tasks = [self._load_model(name, path) for name, path in self.model_configs.items()]
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
            logger.info(f"Loaded model '{model_name}' on device '{self.device_string}'.")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
            self.models[model_name] = None

    def get_embeddings(self, texts: List[str], model_name: str = "semantic") -> np.ndarray:
        """Generates embeddings using a specified model, with fallback for failures."""
        model = self.models.get(model_name)
        if not isinstance(model, SentenceTransformer) or model is None:
            logger.error(f"Embedding model '{model_name}' not available or is not a SentenceTransformer. Returning zero embeddings.")
            # Default to a common embedding dimension (e.g., MiniLM has 384) if model fails to load
            return np.zeros((len(texts), 384), dtype=np.float32)

        embeddings: np.ndarray = model.encode(texts, convert_to_numpy=True, show_progress_bar=False) # type: ignore
        return embeddings

    def rerank_documents(self, query: str, documents_content: List[str], top_k: int) -> List[Tuple[int, float]]:
        """
        Reranks documents using a CrossEncoder model for higher precision.
        `documents_content` is a list of strings (document texts).
        """
        model = self.models.get("cross_encoder")
        if not isinstance(model, CrossEncoder) or model is None:
            logger.warning("Cross-encoder model not available for reranking. Returning unreranked documents (initial order with dummy scores).")
            return [(i, 1.0) for i in range(min(top_k, len(documents_content)))]

        pairs: List[Tuple[str, str]] = [(query, doc_content) for doc_content in documents_content]
        scores: np.ndarray = model.predict(pairs) # type: ignore
        ranked_indices: np.ndarray = np.argsort(scores)[::-1] # Indices of documents in descending order of score

        return [(int(idx), float(scores[idx])) for idx in ranked_indices[:top_k]]


class DocumentGraphBuilder:
    """Builds and leverages knowledge graphs from documents for enhanced retrieval/context."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_extractor: Optional[spacy.language.Language] = None
        self.is_initialized: bool = False
        self._parent_rag_pipeline: Optional[AdvancedRAGPipeline] = None # For accessing `documents` store

    async def initialize(self):
        """Initializes NLP models for graph construction."""
        try:
            loop = asyncio.get_running_loop()
            def load_spacy_sync() -> spacy.language.Language:
                try:
                    return spacy.load("en_core_web_sm")
                except OSError:
                    # Attempt to download if not found, but raise if download fails or isn't possible
                    # In a production environment, ensure models are pre-installed
                    logging.info("spaCy 'en_core_web_sm' model not found. Attempting to download via `spacy.cli.download`.")
                    # spaCy.cli.download("en_core_web_sm") # Uncomment if automatic download is desired in prod
                    return spacy.load("en_core_web_sm") # Try loading again
            self.entity_extractor = await loop.run_in_executor(None, load_spacy_sync)
            self.is_initialized = True
            logger.info("DocumentGraphBuilder initialized with spaCy 'en_core_web_sm'.")
        except Exception as e:
            logger.error("Failed to initialize DocumentGraphBuilder. Graph features will be limited.", error=str(e), exc_info=True)
            self.entity_extractor = None
            self.is_initialized = False


    def build_document_graph(self, documents: List[Document]) -> nx.DiGraph:
        """Builds a knowledge graph from a list of LangChain Documents."""
        self.graph.clear() # Clear existing graph for new build

        if not self.is_initialized or self.entity_extractor is None:
            logger.warning("DocumentGraphBuilder not initialized. Returning empty graph.")
            return self.graph

        for i, doc in enumerate(documents):
            # Create a unique and stable document ID for the graph node
            source_file_clean = doc.metadata.get('source_file', f'unknown_file_{i}').replace('.', '_').replace('/', '_').replace('\\', '_')
            page_number_clean = str(doc.metadata.get('page_number', 'N_A'))
            chunk_id_clean = str(doc.metadata.get('chunk_id', f'chunk_{i}')).replace('-', '_')
            doc_id = f"doc_{source_file_clean}_page{page_number_clean}_chunk{chunk_id_clean}"

            self.graph.add_node(doc_id, type="document", content_snippet=doc.page_content[:150], metadata=doc.metadata)

            entities = self._extract_entities(doc.page_content)
            for entity in entities:
                # Create a unique entity node ID
                entity_text_clean = entity['text'].lower().replace(' ', '_').replace('.', '').replace(',', '')
                entity_label_clean = entity['label'].lower()
                entity_node_id = f"entity_{entity_text_clean}_{entity_label_clean}"

                if not self.graph.has_node(entity_node_id):
                    self.graph.add_node(entity_node_id, type="entity", label=entity["label"], text=entity["text"])
                self.graph.add_edge(doc_id, entity_node_id, relation="contains_entity")

        # Add relationships between documents based on shared entities
        doc_nodes = [n for n, data in self.graph.nodes(data=True) if data['type'] == 'document']
        for i, doc1_id in enumerate(doc_nodes):
            for doc2_id in doc_nodes[i+1:]:
                common_entities = self._find_common_entities(doc1_id, doc2_id)
                if common_entities:
                    self.graph.add_edge(doc1_id, doc2_id, relation="shares_entities", entities=list(common_entities))
                    self.graph.add_edge(doc2_id, doc1_id, relation="shares_entities", entities=list(common_entities))

        logger.info(f"Document graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self.graph

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extracts named entities from text using spaCy."""
        if self.entity_extractor is None:
            return []
        # Process a truncated text to avoid excessively long NLP processing for very large chunks
        doc = self.entity_extractor(text[:1_000_000]) # Limit input size for spaCy
        entities: List[Dict[str, str]] = []
        relevant_labels = ["PERSON", "ORG", "GPE", "LOC", "DATE", "LAW", "NORP", "EVENT", "FAC", "MONEY", "PERCENT", "PRODUCT", "WORK_OF_ART"]
        for ent in doc.ents:
            if ent.label_ in relevant_labels:
                entities.append({"text": ent.text, "label": ent.label_})
        return entities

    def _find_common_entities(self, doc1_id: str, doc2_id: str) -> Set[str]:
        """Finds common entity node IDs between two document nodes in the graph."""
        if doc1_id not in self.graph or doc2_id not in self.graph:
            return set()
        entities1 = {u for v, u in self.graph.edges(doc1_id) if self.graph.nodes[u].get('type') == 'entity'}
        entities2 = {u for v, u in self.graph.edges(doc2_id) if self.graph.nodes[u].get('type') == 'entity'}
        return entities1.intersection(entities2)

    def find_related_documents_from_graph(self, query_entity_texts: List[str], top_k: int = 5) -> List[Document]:
        """
        Finds LangChain Document objects related to query entities using graph traversal.
        Requires a reference to the main RAG pipeline's 'documents' store.
        """
        if not self.graph or not self.is_initialized or self._parent_rag_pipeline is None or not self._parent_rag_pipeline.documents:
            logger.warning("Graph or RAG pipeline documents not available for graph-based retrieval in DocumentGraphBuilder.")
            return []

        related_doc_ids_set: Set[str] = set()

        # Find graph entity nodes matching query entity texts (case-insensitive)
        query_entity_node_ids: Set[str] = set()
        for q_entity_text in query_entity_texts:
            for node_id, node_data in self.graph.nodes(data=True):
                if node_data.get("type") == "entity" and node_data.get("text", "").lower() == q_entity_text.lower():
                    query_entity_node_ids.add(node_id)

        # Traverse from entity nodes to connected document nodes
        for entity_node_id in query_entity_node_ids:
            if entity_node_id in self.graph:
                # Get documents that 'contain' this entity (predecessors in A->B where A=doc, B=entity)
                for doc_node_id in self.graph.predecessors(entity_node_id):
                    if self.graph.nodes[doc_node_id].get("type") == "document":
                        related_doc_ids_set.add(doc_node_id)

                # Find other documents that share common entities with documents found above,
                # or entities that are somehow related to the query entity (more complex traversal)
                # For simplicity, let's just do one hop for now.
                # A more complex traversal would explore paths like (Query Entity) -> (Another Entity) -> (Document)
                for neighbor_entity_id in self.graph.neighbors(entity_node_id):
                    if self.graph.nodes[neighbor_entity_id].get("type") == "entity":
                        for doc_node_id_from_neighbor in self.graph.predecessors(neighbor_entity_id):
                            if self.graph.nodes[doc_node_id_from_neighbor].get("type") == "document":
                                related_doc_ids_set.add(doc_node_id_from_neighbor)


        # Map graph doc_ids back to LangChain Document objects from the pipeline's main store
        doc_id_to_langchain_doc: Dict[str, Document] = {}
        for i, doc in enumerate(self._parent_rag_pipeline.documents):
            # Reconstruct the doc_id used during graph building to match
            source_file_clean = doc.metadata.get('source_file', f'unknown_file_{i}').replace('.', '_').replace('/', '_').replace('\\', '_')
            page_number_clean = str(doc.metadata.get('page_number', 'N_A'))
            chunk_id_clean = str(doc.metadata.get('chunk_id', f'chunk_{i}')).replace('-', '_')
            graph_doc_id = f"doc_{source_file_clean}_page{page_number_clean}_chunk{chunk_id_clean}"
            doc_id_to_langchain_doc[graph_doc_id] = doc

        retrieved_documents: List[Document] = []
        for doc_graph_id in list(related_doc_ids_set):
            if doc_graph_id in doc_id_to_langchain_doc:
                retrieved_documents.append(doc_id_to_langchain_doc[doc_graph_id])
            if len(retrieved_documents) >= top_k:
                break # Limit to top_k graph-related documents

        return retrieved_documents


class HybridRetriever:
    """Advanced hybrid retrieval combining multiple strategies."""
    def __init__(self, embedding_manager: AdvancedEmbeddingManager, config: RetrievalConfig):
        self.embedding_manager = embedding_manager
        self.config = config
        self.vector_store: Optional[faiss.IndexFlatIP] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.documents_store: List[Document] = [] # The full list of indexed documents
        self.document_clusters: Dict[int, List[int]] = {} # Map cluster ID to list of document indices
        self.cross_references: Dict[int, List[int]] = {} # Map document index to list of related document indices

    async def initialize(self, documents: List[Document]):
        """Initializes all retrieval components from a list of documents."""
        logger.info("Initializing hybrid retriever...")
        if not documents:
            logger.warning("No documents provided to initialize hybrid retriever. Retrieval will be empty.")
            self.documents_store = []
            self.vector_store = None
            self.bm25_retriever = None
            return

        self.documents_store = documents
        texts = [doc.page_content for doc in documents]

        # Build FAISS vector store for semantic search
        try:
            embeddings: np.ndarray = self.embedding_manager.get_embeddings(texts, "semantic")
            if embeddings.size == 0 or embeddings.ndim == 1:
                raise ValueError("Embedding manager returned empty or invalid embeddings.")

            dimension: int = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatIP(dimension) # Inner Product for cosine similarity with L2 normalized vectors
            faiss.normalize_L2(embeddings) # Normalize embeddings for Inner Product search
            self.vector_store.add(embeddings.astype('float32'))
            logger.info(f"FAISS vector store built with {len(documents)} documents, dimension {dimension}.")
        except Exception as e:
            logger.error(f"Failed to build FAISS vector store: {e}", exc_info=True)
            self.vector_store = None

        # Build BM25 retriever for keyword/lexical search
        try:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = self.config.max_initial_retrieval_docs # Number of docs to retrieve initially
            logger.info("BM25 retriever built.")
        except Exception as e:
            logger.error(f"Failed to build BM25 retriever: {e}", exc_info=True)
            self.bm25_retriever = None

        # Build document clusters asynchronously (CPU-bound, use executor)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._build_document_clusters, documents)
        except Exception as e:
            logger.warning(f"Failed to build document clusters: {e}", exc_info=True)

        # Build cross-references asynchronously (CPU-bound, use executor)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._build_cross_references, documents)
        except Exception as e:
            logger.warning(f"Failed to build cross-references: {e}", exc_info=True)

        logger.info("Hybrid retriever initialization attempt complete.")

    def _build_document_clusters(self, documents: List[Document]):
        """Builds semantic clusters of documents using DBSCAN."""
        if len(documents) < self.config.semantic_clustering_min_samples:
            self.document_clusters = {}
            logger.debug(f"Skipping DBSCAN clustering: Not enough documents ({len(documents)}) for min_samples ({self.config.semantic_clustering_min_samples}).")
            return

        texts = [doc.page_content for doc in documents]
        embeddings: np.ndarray = self.embedding_manager.get_embeddings(texts, "semantic")

        if embeddings.size == 0 or embeddings.ndim == 1 or np.all(embeddings == 0):
            logger.warning("Embeddings are empty, 1D, or all zeros. Skipping DBSCAN clustering.")
            self.document_clusters = {}
            return

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=self.config.semantic_clustering_eps, min_samples=self.config.semantic_clustering_min_samples, metric='cosine')
        cluster_labels: np.ndarray = clustering.fit_predict(embeddings) # type: ignore

        self.document_clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in self.document_clusters:
                self.document_clusters[label] = []
            self.document_clusters[label].append(i)

        logger.info(f"Created {len(self.document_clusters)} document clusters (including noise if any).")

    def _build_cross_references(self, documents: List[Document]):
        """Builds cross-reference mapping between documents based on shared keywords/entities."""
        if not documents:
            self.cross_references = {}
            return

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform([doc.page_content for doc in documents])
        except ValueError as e:
            logger.warning(f"TfidfVectorizer failed for cross-references: {e}. Skipping.")
            self.cross_references = {}
            return

        cosine_sim_matrix: np.ndarray = cosine_similarity(tfidf_matrix)

        self.cross_references = {}
        for i in range(len(documents)):
            self.cross_references[i] = []
            for j in range(i + 1, len(documents)):
                # Use a configurable threshold for cosine similarity
                if cosine_sim_matrix[i, j] > self.config.min_overlap_for_cross_ref:
                    self.cross_references[i].append(j)
                    self.cross_references[j].append(i) # Bidirectional link

        logger.info(f"Built cross-references for {len(documents)} documents.")


    async def retrieve(self, query_context: QueryContext) -> RetrievalResult:
        """Performs hybrid retrieval using multiple strategies, including multi-query and reranking."""
        start_time = datetime.now()

        if self.vector_store is None or self.bm25_retriever is None or not self.documents_store:
            logger.error("HybridRetriever not fully initialized or no documents loaded. Returning empty results.")
            return RetrievalResult(
                [], [], 0.0, "failed", (datetime.now() - start_time).total_seconds(),
                query_analysis=query_context.__dict__ # Pass the original query context
            )

        # Multi-query expansion (using queries from the analyzed context)
        queries_to_search: List[str] = [query_context.original_query]
        if self.config.enable_multi_query_expansion and query_context.expanded_queries:
            queries_to_search.extend(query_context.expanded_queries)

        all_retrieved_docs_map: Dict[str, Document] = {} # Map doc_content to Document to avoid duplicates
        all_retrieved_scores_map: Dict[str, float] = {} # Map doc_content to its highest score

        for q_str in queries_to_search:
            # BM25 Search
            if self.bm25_retriever:
                try:
                    bm25_results: List[Document] = self.bm25_retriever.get_relevant_documents(q_str) # type: ignore
                    for doc in bm25_results:
                        doc_key: str = doc.page_content # Use content as key for uniqueness
                        score: float = self.config.bm25_weight # BM25 doesn't return scores directly, use weight
                        if doc_key not in all_retrieved_docs_map or all_retrieved_scores_map[doc_key] < score:
                            all_retrieved_docs_map[doc_key] = doc
                            all_retrieved_scores_map[doc_key] = score
                except Exception as e:
                    logger.warning(f"BM25 search failed for query '{q_str[:50]}...': {e}")

            # Vector Search (FAISS)
            if self.vector_store:
                try:
                    query_embedding: np.ndarray = self.embedding_manager.get_embeddings([q_str], "semantic")[0]
                    query_embedding_reshaped: np.ndarray = np.expand_dims(query_embedding, axis=0).astype('float32')
                    faiss.normalize_L2(query_embedding_reshaped)

                    distances: np.ndarray
                    indices: np.ndarray
                    distances, indices = self.vector_store.search(query_embedding_reshaped, self.config.max_initial_retrieval_docs) # type: ignore

                    for i, idx in enumerate(indices[0]):
                        if 0 <= idx < len(self.documents_store): # Ensure index is within bounds
                            doc: Document = self.documents_store[idx]
                            score: float = float(distances[0][i]) # FAISS IP returns scores

                            weighted_score: float = score * self.config.vector_weight
                            doc_key = doc.page_content
                            if doc_key not in all_retrieved_docs_map or all_retrieved_scores_map[doc_key] < weighted_score:
                                all_retrieved_docs_map[doc_key] = doc
                                all_retrieved_scores_map[doc_key] = weighted_score
                        else:
                            logger.warning(f"FAISS returned out-of-bounds index: {idx}")
                except Exception as e:
                    logger.warning(f"Vector search failed for query '{q_str[:50]}...': {e}")

        # Combine and sort initial retrieval results
        initial_retrieved_list: List[Tuple[Document, float]] = sorted(
            [(doc, all_retrieved_scores_map[doc.page_content]) for doc in all_retrieved_docs_map.values()],
            key=lambda x: x[1], reverse=True
        )[:self.config.max_initial_retrieval_docs]

        final_docs_for_llm: List[Document] = [doc for doc, _ in initial_retrieved_list]
        relevance_scores: List[float] = [score for _, score in initial_retrieved_list]

        # Cross-encoder Reranking for higher precision
        if self.config.enable_cross_encoder_reranking and final_docs_for_llm:
            doc_contents: List[str] = [doc.page_content for doc in final_docs_for_llm]
            reranked_tuples: List[Tuple[int, float]] = self.embedding_manager.rerank_documents(
                query_context.original_query,
                doc_contents,
                self.config.max_reranked_docs
            )

            # Reorder documents based on reranker scores
            reranked_final_docs: List[Document] = []
            reranked_relevance_scores: List[float] = []
            for idx, score in reranked_tuples:
                if idx < len(final_docs_for_llm): # Ensure index is valid
                    doc = final_docs_for_llm[idx]
                    # Update metadata with reranked score for downstream use (e.g., citations)
                    doc.metadata["relevance_score"] = score
                    reranked_final_docs.append(doc)
                    reranked_relevance_scores.append(score)
            # Use reranked results
            final_docs_for_llm = reranked_final_docs
            relevance_scores = reranked_relevance_scores

        # Document Graph Analysis for additional context/related documents
        if self.config.enable_document_graph_analysis and self.embedding_manager._parent_rag_pipeline is not None \
           and self.embedding_manager._parent_rag_pipeline.document_graph_builder.is_initialized \
           and query_context.entity_extraction:

            query_entity_texts: List[str] = [ent['text'] for ent in query_context.entity_extraction if 'text' in ent]
            if query_entity_texts:
                graph_related_docs: List[Document] = self.embedding_manager._parent_rag_pipeline.document_graph_builder.find_related_documents_from_graph(
                    query_entity_texts, self.config.max_reranked_docs // 2 # Retrieve a limited number of graph docs
                )

                # Add graph-related documents to the final list, avoiding duplicates
                current_doc_contents: Set[str] = {d.page_content for d in final_docs_for_llm}
                for g_doc in graph_related_docs:
                    if g_doc.page_content not in current_doc_contents:
                        final_docs_for_llm.append(g_doc)
                        relevance_scores.append(0.3) # Assign a default relevance score for graph-added docs
                        current_doc_contents.add(g_doc.page_content) # Update for subsequent checks

                # Re-sort and truncate after adding graph documents to ensure top_k is maintained
                combined_docs_with_scores: List[Tuple[Document, float]] = sorted(
                    zip(final_docs_for_llm, relevance_scores),
                    key=lambda x: x[1], reverse=True
                )[:self.config.max_reranked_docs]
                final_docs_for_llm = [d for d, _ in combined_docs_with_scores]
                relevance_scores = [s for _, s in combined_docs_with_scores]

        processing_time: float = (datetime.now() - start_time).total_seconds()

        # Calculate overall confidence score
        if relevance_scores:
            avg_relevance: float = np.mean(relevance_scores)
            num_docs_factor: float = min(len(final_docs_for_llm) / self.config.max_reranked_docs, 1.0)
            confidence_score: float = float(avg_relevance * 0.7 + num_docs_factor * 0.3) # Weighted average
        else:
            confidence_score = 0.0

        return RetrievalResult(
            documents=final_docs_for_llm,
            relevance_scores=relevance_scores,
            confidence_score=confidence_score,
            retrieval_strategy="hybrid_multiquery_reranked_graph_aware",
            processing_time=processing_time,
            query_analysis=query_context.__dict__, # Store the internal RAG QueryContext as dict
            document_clusters=list(self.document_clusters.values()),
            cross_references=self.cross_references,
            legal_citations=self._extract_legal_citations(final_docs_for_llm)
        )

    def _extract_legal_citations(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extracts and formats legal citations from documents' metadata."""
        citations: List[Dict[str, Any]] = []
        for doc in documents:
            citations.append({
                "source_file": doc.metadata.get("source_file", "unknown_source"),
                "page_number": doc.metadata.get("page_number", 1),
                "section_title": doc.metadata.get("section_title", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "relevance_score": doc.metadata.get("relevance_score", 0.0)
            })
        return citations


class AdvancedRAGPipeline:
    """
    Main RAG pipeline orchestrating all components for advanced legal information retrieval.
    This class does NOT create its own instances of external services (CacheManager, QueryProcessor, etc.).
    It relies on dependency injection for these components, which are managed and initialized by the main application.
    """

    def __init__(self,
                 cache_manager: CacheManager,
                 query_analyzer: IntelligentQueryProcessor,
                 document_processor: AdvancedDocumentProcessor,
                 model_orchestrator: ProductionModelOrchestrator,
                 monitoring_engine: ProductionMonitoringEngine):

        self.cache_manager = cache_manager
        self.query_analyzer = query_analyzer
        self.document_processor = document_processor
        self.model_orchestrator = model_orchestrator
        self.monitoring_engine = monitoring_engine

        self.config = RetrievalConfig() # Internal RAG configuration
        self.embedding_manager = AdvancedEmbeddingManager() # Internal to RAG
        self.embedding_manager._parent_rag_pipeline = self # Link back for graph builder

        self.document_graph_builder = DocumentGraphBuilder() # Internal to RAG
        self.document_graph_builder._parent_rag_pipeline = self # Link back for documents access

        self.hybrid_retriever: Optional[HybridRetriever] = None # Internal to RAG
        self.documents: List[Document] = [] # The knowledge base for the RAG pipeline
        self.is_initialized: bool = False

    async def initialize(self):
        """
        Initializes the RAG pipeline's internal components.
        It does NOT initialize the injected external components (cache, query_analyzer, etc.),
        as they are expected to be initialized by the main application orchestrator.
        """
        logger.info("Initializing Advanced RAG Pipeline (internal components only)...")
        try:
            # Initialize core RAG components
            await self.embedding_manager.initialize()
            await self.document_graph_builder.initialize()

            # Load and process documents to build the RAG's knowledge base
            await self._load_documents()

            # Initialize hybrid retriever with the processed documents
            self.hybrid_retriever = HybridRetriever(self.embedding_manager, self.config)
            await self.hybrid_retriever.initialize(self.documents)

            self.is_initialized = True
            logger.info("Advanced RAG Pipeline (internal components) initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Advanced RAG Pipeline", error=str(e), exc_info=True)
            self.is_initialized = False
            raise # Re-raise to indicate a critical startup failure

    async def _load_documents(self):
        """
        Loads and processes legal documents from the specified directory
        using the injected AdvancedDocumentProcessor.
        """
        legal_docs_path: Path = Path("legal_docs")
        if not legal_docs_path.exists() or not any(legal_docs_path.iterdir()):
            logger.warning("`legal_docs` directory not found or is empty. RAG pipeline will have no knowledge base.")
            self.documents = []
            return

        logger.info(f"Scanning and processing documents from: {legal_docs_path.resolve()}")

        supported_extensions: List[str] = ['.pdf', '.docx', '.txt', '.md', '.xlsx', '.zip', '.rar', '.7z']
        doc_paths: List[Path] = [p for p in legal_docs_path.rglob('*') if p.suffix.lower() in supported_extensions]

        if not doc_paths:
            logger.warning("No supported documents found in `legal_docs`. RAG pipeline will have no knowledge base.")
            self.documents = []
            return

        # Use the injected document processor for batch processing
        processed_docs_results: List[ProcessedDocument] = await self.document_processor.batch_process(
            [str(p) for p in doc_paths]
        )

        all_chunks: List[Document] = []
        for proc_doc in processed_docs_results:
            if proc_doc.metadata.status == ProcessingStatus.COMPLETED and proc_doc.chunks:
                for chunk in proc_doc.chunks:
                    # Create LangChain Document objects from processed chunks
                    langchain_doc = Document(
                        page_content=chunk.content,
                        metadata={
                            "source_file": proc_doc.metadata.file_name,
                            "page_number": chunk.page_number,
                            "section_title": chunk.section_title,
                            "chunk_id": chunk.chunk_id,
                            "file_hash": proc_doc.metadata.file_hash,
                            "doc_type": proc_doc.metadata.document_type.value,
                            "language": proc_doc.metadata.language,
                            "summary": proc_doc.metadata.summary,
                            "keywords": proc_doc.metadata.keywords,
                            "entities": proc_doc.metadata.entities, # Entities from document processing
                            "pii_detected_count": len(proc_doc.metadata.pii_detected),
                            "full_text_length": len(proc_doc.full_text)
                        }
                    )
                    all_chunks.append(langchain_doc)
            elif proc_doc.metadata.status == ProcessingStatus.FAILED:
                logger.error(f"Document processing failed for {proc_doc.metadata.file_name}: {proc_doc.metadata.error_message}")
            elif proc_doc.metadata.status == ProcessingStatus.SECURITY_RISK:
                 logger.warning(f"Document {proc_doc.metadata.file_name} was flagged as security risk and skipped.")

        self.documents = all_chunks
        logger.info(f"Successfully loaded and chunked {len(all_chunks)} document segments from {len(doc_paths)} files.")

        # Build document graph if enabled and documents are available
        if self.config.enable_document_graph_analysis and self.document_graph_builder.is_initialized and self.documents:
            self.document_graph_builder.build_document_graph(self.documents)


    async def enhanced_query(self, query: str, language: str, processed_query_obj: ProcessedQuery, query_id: str) -> Dict[str, Any]:
        """
        Processes an enhanced query through the RAG pipeline.
        It receives an already analyzed `ProcessedQuery` object from the `IntelligentQueryProcessor`.
        """
        if not self.is_initialized or self.hybrid_retriever is None:
            raise RuntimeError("RAG Pipeline not initialized. Cannot process query.")

        start_time = datetime.now()

        # Map the IntelligentQueryProcessor's ProcessedQuery to RAG's internal QueryContext
        # This ensures RAG gets consistent, structured query metadata
        
        # Mapping for complexity score (Enum to float)
        complexity_map: Dict[QueryComplexity, float] = {
            QueryComplexity.SIMPLE: 0.25,
            QueryComplexity.MODERATE: 0.5,
            QueryComplexity.COMPLEX: 0.75,
            QueryComplexity.VERY_COMPLEX: 1.0
        }
        
        # Mapping entities from IntelligentQueryProcessor.LegalEntity to simple dict
        rag_entities: List[Dict[str, str]] = [{'text': ent.text, 'label': ent.entity_type} for ent in processed_query_obj.entities]

        query_context_obj = QueryContext(
            original_query=processed_query_obj.original_query,
            # Expanded queries expected from IQP's metadata or a dedicated field
            expanded_queries=processed_query_obj.metadata.get("expanded_queries", []) if processed_query_obj.metadata else [],
            intent_classification={processed_query_obj.intent.value: processed_query_obj.intent_confidence},
            entity_extraction=rag_entities,
            legal_domain=processed_query_obj.metadata.get("legal_domain", "general") if processed_query_obj.metadata else "general",
            complexity_score=complexity_map.get(processed_query_obj.complexity, 0.5), # Default to moderate if mapping fails
            urgency_level=processed_query_obj.emotional_tone.value, # Enum value (str)
            user_expertise=processed_query_obj.context.user_expertise if processed_query_obj.context and hasattr(processed_query_obj.context, 'user_expertise') else "general",
            session_context=processed_query_obj.context.session_context if processed_query_obj.context and hasattr(processed_query_obj.context, 'session_context') else {},
            temporal_context=processed_query_obj.context.temporal_context if processed_query_obj.context and hasattr(processed_query_obj.context, 'temporal_context') else {}
        )

        try:
            # Step 1: Hybrid Retrieval using the constructed QueryContext
            retrieval_result = await self.hybrid_retriever.retrieve(query_context_obj)

            # Step 2: LLM Response Generation using the retrieved documents and query analysis
            response: Dict[str, Any] = await self.model_orchestrator.generate_response(
                query=query,
                context_documents=retrieval_result.documents,
                language=language, # The language detected by IQP or from request
                query_analysis=processed_query_obj.__dict__ # Pass the full processed query object as dict
            )

            # Final result assembly
            final_result: Dict[str, Any] = {
                "answer": response["answer"],
                "source_documents": retrieval_result.documents,
                "confidence_score": retrieval_result.confidence_score,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "retrieval_metadata": {
                    "strategy": retrieval_result.retrieval_strategy,
                    "legal_citations": retrieval_result.legal_citations,
                    "document_clusters": retrieval_result.document_clusters,
                    "cross_references": retrieval_result.cross_references
                }
            }

            # Log metrics via the injected monitoring engine
            await self.monitoring_engine.log_query_metrics({
                "query_id": query_id,
                "original_query": query,
                "processing_time_rag": final_result["processing_time"], # RAG specific time
                "documents_retrieved": len(retrieval_result.documents),
                "confidence_score": final_result["confidence_score"],
                "retrieval_strategy": final_result["retrieval_metadata"]["strategy"],
                "intent": processed_query_obj.intent.value,
                "language": processed_query_obj.language.value,
                "complexity": processed_query_obj.complexity.value,
                "emotional_tone": processed_query_obj.emotional_tone.value
            })

            return final_result

        except Exception as e:
            logger.error("Error in enhanced query processing within RAG pipeline", query_id=query_id, error=str(e), exc_info=True)
            raise # Re-raise to be handled by the calling component (e.g., FastAPI endpoint)

    async def health_check(self) -> bool:
        """
        Checks the health of RAG pipeline's internal components.
        It also delegates health checks to the injected external components.
        """
        if not self.is_initialized:
            logger.warning("RAG Pipeline is not initialized.")
            return False

        health_status: bool = True

        # 1. Check internal RAG components
        if not self.embedding_manager.models or any(m is None for m in self.embedding_manager.models.values()):
            logger.error("RAG: Embedding manager models not fully loaded.")
            health_status = False

        if self.hybrid_retriever is None or not self.hybrid_retriever.documents_store:
            logger.error("RAG: Hybrid retriever not initialized or has no documents.")
            health_status = False

        if not self.document_graph_builder.is_initialized:
            logger.warning("RAG: Document graph builder not initialized. Graph features might be missing.")

        # 2. Delegate health checks to injected components
        try:
            if not await self.cache_manager.health_check():
                logger.error("RAG: Injected CacheManager failed health check.")
                health_status = False
        except Exception as e:
            logger.error(f"RAG: Error checking Injected CacheManager health: {e}", exc_info=True)
            health_status = False

        try:
            if not await self.query_analyzer.health_check():
                logger.error("RAG: Injected IntelligentQueryProcessor failed health check.")
                health_status = False
        except Exception as e:
            logger.error(f"RAG: Error checking Injected IntelligentQueryProcessor health: {e}", exc_info=True)
            health_status = False

        try:
            if not await self.document_processor.health_check():
                logger.error("RAG: Injected AdvancedDocumentProcessor failed health check.")
                health_status = False
        except Exception as e:
            logger.error(f"RAG: Error checking Injected AdvancedDocumentProcessor health: {e}", exc_info=True)
            health_status = False

        try:
            if not await self.model_orchestrator.health_check():
                logger.error("RAG: Injected ProductionModelOrchestrator failed health check.")
                health_status = False
        except Exception as e:
            logger.error(f"RAG: Error checking Injected ProductionModelOrchestrator health: {e}", exc_info=True)
            health_status = False

        try:
            if not await self.monitoring_engine.health_check():
                logger.error("RAG: Injected ProductionMonitoringEngine failed health check.")
                health_status = False
        except Exception as e:
            logger.error(f"RAG: Error checking Injected ProductionMonitoringEngine health: {e}", exc_info=True)
            health_status = False

        return health_status