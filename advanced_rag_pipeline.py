"""
Advanced RAG Pipeline with Multi-Modal Retrieval, Hybrid Search, and Intelligent Ranking
Implements state-of-the-art retrieval-augmented generation with sophisticated document processing.
Version: 2.3 (Enterprise Edition - Comprehensive Bug Fixes and Enhanced Robustness)
"""

import asyncio
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple, Union, Set 
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import os
import shutil 


import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.schema import Document

from langchain_community.retrievers import BM25Retriever
import spacy
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN 
import structlog



from advanced_document_processor import AdvancedDocumentProcessor, ProcessedDocument, ProcessingStatus 






logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = structlog.get_logger(__name__)

@dataclass
class RetrievalConfig:
    """Centralized configuration for advanced retrieval strategies."""
    max_initial_retrieval_docs: int = 50 
    max_reranked_docs: int = 10          
    bm25_weight: float = 0.4              
    vector_weight: float = 0.6            
    min_overlap_for_cross_ref: int = 10   
    semantic_clustering_eps: float = 0.3  
    semantic_clustering_min_samples: int = 2 
    enable_cross_encoder_reranking: bool = True
    enable_multi_query_expansion: bool = True
    enable_contextual_compression: bool = False 
    enable_document_graph_analysis: bool = True
    
@dataclass
class QueryContext:
    """Enhanced query context with multi-dimensional analysis from IntelligentQueryProcessor."""
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    intent_classification: Dict[str, float] = field(default_factory=dict)
    entity_extraction: List[Dict[str, str]] = field(default_factory=list) 
    legal_domain: str = ""
    complexity_score: float = 0.0
    urgency_level: str = "normal"
    user_expertise: str = "general"
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
    query_analysis: Dict[str, Any] = field(default_factory=dict)
    document_clusters: List[List[int]] = field(default_factory=list)
    cross_references: Dict[int, List[int]] = field(default_factory=list) 
    legal_citations: List[Dict[str, Any]] = field(default_factory=list)






class DummyCacheManager:
    """Mocks caching functionality for isolated RAG pipeline testing."""
    async def get_cached_response(self, *args, **kwargs): return None
    async def cache_response(self, *args, **kwargs): pass
    async def initialize(self): logger.info("DummyCacheManager initialized.")
    async def close(self): logger.info("DummyCacheManager closed.")
    async def health_check(self): return True

class DummyQueryAnalyzer:
    """Mocks query analysis for isolated RAG pipeline testing."""
    async def comprehensive_analyze(self, query: str, **kwargs) -> Dict[str, Any]:
        
        return {
            "original_query": query,
            "expanded_queries": [query, f"{query} legal aspects", f"case law for {query}"],
            "intent_classification": {"legal_information": 0.9},
            "entity_extraction": [{"text": "Rwanda", "label": "GPE"}, {"text": "Mr. Smith", "label": "PERSON"}], 
            "legal_domain": "general",
            "complexity_score": 0.5,
            "urgency_level": "normal",
            "session_context": {},
            "temporal_context": {}
        }
    async def initialize(self): logger.info("DummyQueryAnalyzer initialized.")
    async def health_check(self): return True

class DummyModelOrchestrator:
    """Mocks LLM response generation for isolated RAG pipeline testing."""
    async def generate_response(self, query: str, context_documents: List[Document], language: str, query_analysis: Dict) -> Dict[str, Any]:
        context_snippet = "\n".join([doc.page_content[:150] + "..." for doc in context_documents])
        answer = f"Hello! Based on the {len(context_documents)} retrieved documents for '{query}' (lang: {language}), here's a synthesized answer:\n\n{context_snippet}"
        return {"answer": answer}
    async def initialize(self): logger.info("DummyModelOrchestrator initialized.")
    async def health_check(self): return True

class DummyMonitoringEngine:
    """Mocks monitoring and metrics logging."""
    async def log_query_metrics(self, *args, **kwargs): pass
    async def initialize(self): logger.info("DummyMonitoringEngine initialized.")
    async def health_check(self): return True





class AdvancedEmbeddingManager:
    """Manages multiple embedding models for different use cases."""
    def __init__(self):
        self.models: Dict[str, Optional[Union[SentenceTransformer, CrossEncoder]]] = {} 
        
        self.device_string = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_configs = {
            "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "semantic": "sentence-transformers/all-MiniLM-L6-v2"
        }
        self._parent_rag_pipeline: Optional['AdvancedRAGPipeline'] = None 

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
            
            
            return np.zeros((len(texts), 384), dtype=np.float32) 
        
        
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
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
        scores = model.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1] 
        
        return [(int(idx), float(scores[idx])) for idx in ranked_indices[:top_k]]





class DocumentGraphBuilder:
    """Builds and leverages knowledge graphs from documents for enhanced retrieval/context."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_extractor: Optional[spacy.language.Language] = None 
        self.is_initialized = False
        self._parent_rag_pipeline: Optional['AdvancedRAGPipeline'] = None 

    async def initialize(self):
        """Initializes NLP models for graph construction."""
        try:
            loop = asyncio.get_running_loop()
            def load_spacy_sync() -> spacy.language.Language: 
                try:
                    return spacy.load("en_core_web_sm")
                except OSError:
                    logging.info("spaCy 'en_core_web_sm' model not found. Attempting to download.")
                    
                    
                    
                    
                    raise RuntimeError("spaCy 'en_core_web_sm' model not found. Please run 'python -m spacy download en_core_web_sm'.")

            self.entity_extractor = await loop.run_in_executor(None, load_spacy_sync)
            self.is_initialized = True
            logger.info("DocumentGraphBuilder initialized with spaCy 'en_core_web_sm'.")
        except Exception as e:
            logger.error("Failed to initialize DocumentGraphBuilder. Graph features will be limited.", error=str(e), exc_info=True)
            self.entity_extractor = None

    def build_document_graph(self, documents: List[Document]) -> nx.DiGraph:
        """Builds a knowledge graph from a list of LangChain Documents."""
        self.graph.clear() 
        
        if not self.is_initialized or self.entity_extractor is None: 
            logger.warning("DocumentGraphBuilder not initialized. Returning empty graph.")
            return self.graph

        for i, doc in enumerate(documents):
            
            source_file_clean = doc.metadata.get('source_file', 'unknown').replace('.', '_').replace('/', '_').replace('\\', '_')
            page_number_clean = str(doc.metadata.get('page_number', 'N/A'))
            chunk_id_clean = str(doc.metadata.get('chunk_id', i)).replace('-', '_')
            doc_id = f"doc_{source_file_clean}_page{page_number_clean}_chunk{chunk_id_clean}"

            self.graph.add_node(doc_id, type="document", content_snippet=doc.page_content[:150], metadata=doc.metadata)
            
            entities = self._extract_entities(doc.page_content)
            for entity in entities:
                
                entity_text_clean = entity['text'].lower().replace(' ', '_').replace('.', '')
                entity_label_clean = entity['label'].lower()
                entity_node_id = f"entity_{entity_text_clean}_{entity_label_clean}"
                
                if not self.graph.has_node(entity_node_id):
                    self.graph.add_node(entity_node_id, type="entity", label=entity["label"], text=entity["text"])
                self.graph.add_edge(doc_id, entity_node_id, relation="contains_entity")
        
        
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
        if self.entity_extractor is None: return [] 
        doc = self.entity_extractor(text)
        entities = []
        relevant_labels = ["PERSON", "ORG", "GPE", "LOC", "DATE", "LAW", "NORP", "EVENT", "FAC", "MONEY", "PERCENT"] 
        for ent in doc.ents:
            if ent.label_ in relevant_labels:
                entities.append({"text": ent.text, "label": ent.label_})
        return entities
    
    def _find_common_entities(self, doc1_id: str, doc2_id: str) -> Set[str]:
        """Finds common entity node IDs (not just text) between two document nodes in the graph."""
        if doc1_id not in self.graph or doc2_id not in self.graph:
            return set()
        entities1 = {u for v, u in self.graph.edges(doc1_id) if self.graph.nodes[u]['type'] == 'entity'}
        entities2 = {u for v, u in self.graph.edges(doc2_id) if self.graph.nodes[u]['type'] == 'entity'}
        return entities1.intersection(entities2)

    def find_related_documents_from_graph(self, query_entity_texts: List[str], top_k: int = 5) -> List[Document]:
        """
        Finds LangChain Document objects related to query entities using graph traversal.
        This requires a reference to the main RAG pipeline's 'documents' store.
        """
        if not self.graph or not self.is_initialized or self._parent_rag_pipeline is None or not self._parent_rag_pipeline.documents:
            logger.warning("Graph or RAG pipeline documents not available for graph-based retrieval in DocumentGraphBuilder.")
            return []

        related_doc_ids_set = set()
        
        
        query_entity_node_ids = set()
        for q_entity_text in query_entity_texts:
            
            for node_id, node_data in self.graph.nodes(data=True):
                if node_data.get("type") == "entity" and node_data.get("text", "").lower() == q_entity_text.lower():
                    query_entity_node_ids.add(node_id)
        
        
        for entity_node_id in query_entity_node_ids:
            if entity_node_id in self.graph:
                
                
                for neighbor in self.graph.predecessors(entity_node_id): 
                    if self.graph.nodes[neighbor].get("type") == "document":
                        related_doc_ids_set.add(neighbor)
                
                
                
                for entity_neighbor_id in self.graph.neighbors(entity_node_id): 
                    if self.graph.nodes[entity_neighbor_id].get("type") == 'entity':
                        
                        for doc_connected_to_entity_neighbor in self.graph.predecessors(entity_neighbor_id):
                            if self.graph.nodes[doc_connected_to_entity_neighbor].get("type") == "document":
                                related_doc_ids_set.add(doc_connected_to_entity_neighbor)
        
        
        
        doc_id_to_langchain_doc: Dict[str, Document] = {}
        for i, doc in enumerate(self._parent_rag_pipeline.documents):
            
            source_file_clean = doc.metadata.get('source_file', 'unknown').replace('.', '_').replace('/', '_').replace('\\', '_')
            page_number_clean = str(doc.metadata.get('page_number', 'N/A'))
            chunk_id_clean = str(doc.metadata.get('chunk_id', i)).replace('-', '_')
            graph_doc_id = f"doc_{source_file_clean}_page{page_number_clean}_chunk{chunk_id_clean}"
            doc_id_to_langchain_doc[graph_doc_id] = doc

        retrieved_documents = []
        for doc_graph_id in list(related_doc_ids_set):
            if doc_graph_id in doc_id_to_langchain_doc:
                retrieved_documents.append(doc_id_to_langchain_doc[doc_graph_id])
            if len(retrieved_documents) >= top_k:
                break 
        
        return retrieved_documents






class HybridRetriever:
    """Advanced hybrid retrieval combining multiple strategies."""
    def __init__(self, embedding_manager: AdvancedEmbeddingManager, config: RetrievalConfig):
        self.embedding_manager = embedding_manager
        self.config = config
        self.vector_store: Optional[faiss.IndexFlatIP] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.documents_store: List[Document] = [] 
        self.document_clusters: Dict[int, List[int]] = {} 
        self.cross_references: Dict[int, List[int]] = {} 

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

        
        try:
            embeddings = self.embedding_manager.get_embeddings(texts, "semantic")
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatIP(dimension)  
            faiss.normalize_L2(embeddings)
            self.vector_store.add(embeddings.astype('float32'))
            logger.info(f"FAISS vector store built with {len(documents)} documents, dimension {dimension}.")
        except Exception as e:
            logger.error(f"Failed to build FAISS vector store: {e}", exc_info=True)
            self.vector_store = None

        
        try:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = self.config.max_initial_retrieval_docs
            logger.info("BM25 retriever built.")
        except Exception as e:
            logger.error(f"Failed to build BM25 retriever: {e}", exc_info=True)
            self.bm25_retriever = None
        
        
        try:
            
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._build_document_clusters, documents)
        except Exception as e:
            logger.warning(f"Failed to build document clusters: {e}", exc_info=True)

        
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
            return
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_manager.get_embeddings(texts, "semantic")
        
        
        if embeddings.ndim == 1 and np.all(embeddings == 0):
            logger.warning("Embeddings are zero. Skipping DBSCAN clustering.")
            self.document_clusters = {}
            return

        
        clustering = DBSCAN(eps=self.config.semantic_clustering_eps, min_samples=self.config.semantic_clustering_min_samples, metric='cosine')
        self.document_clusters = {} 
        cluster_labels = clustering.fit_predict(embeddings) 
        
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
        tfidf_matrix = vectorizer.fit_transform([doc.page_content for doc in documents])
        
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
        
        self.cross_references = {}
        for i in range(len(documents)):
            self.cross_references[i] = []
            for j in range(i + 1, len(documents)):
                if cosine_sim_matrix[i, j] > self.config.min_overlap_for_cross_ref: 
                    self.cross_references[i].append(j)
                    self.cross_references[j].append(i) 

        logger.info(f"Built cross-references for {len(documents)} documents.")

    async def retrieve(self, query_context: QueryContext) -> RetrievalResult:
        """Performs hybrid retrieval using multiple strategies, including multi-query and reranking."""
        start_time = datetime.now()
        
        if self.vector_store is None or self.bm25_retriever is None or not self.documents_store: 
            logger.error("HybridRetriever not fully initialized or no documents loaded. Returning empty results.")
            return RetrievalResult(
                [], [], 0.0, "failed", (datetime.now() - start_time).total_seconds(), 
                query_analysis=query_context.__dict__
            )

        
        queries_to_search = [query_context.original_query]
        if self.config.enable_multi_query_expansion and query_context.expanded_queries:
            queries_to_search.extend(query_context.expanded_queries)
        
        all_retrieved_docs_map: Dict[str, Document] = {} 
        all_retrieved_scores_map: Dict[str, float] = {} 

        for q_str in queries_to_search:
            
            if self.bm25_retriever: 
                try:
                    
                    bm25_results = self.bm25_retriever.get_relevant_documents(q_str)
                    for doc in bm25_results:
                        
                        doc_key = doc.page_content 
                        score = self.config.bm25_weight 
                        if doc_key not in all_retrieved_docs_map or all_retrieved_scores_map[doc_key] < score:
                            all_retrieved_docs_map[doc_key] = doc
                            all_retrieved_scores_map[doc_key] = score
                except Exception as e:
                    logger.warning(f"BM25 search failed for query '{q_str[:50]}...': {e}")

            
            if self.vector_store: 
                try:
                    query_embedding = self.embedding_manager.get_embeddings([q_str], "semantic")[0]
                    query_embedding_reshaped = np.expand_dims(query_embedding, axis=0).astype('float32')
                    faiss.normalize_L2(query_embedding_reshaped)
                    
                    
                    distances, indices = self.vector_store.search(query_embedding_reshaped, self.config.max_initial_retrieval_docs) 
                    
                    for i, idx in enumerate(indices[0]):
                        if 0 <= idx < len(self.documents_store): 
                            doc = self.documents_store[idx]
                            score = float(distances[0][i]) 
                            doc_key = doc.page_content
                            
                            weighted_score = score * self.config.vector_weight
                            if doc_key not in all_retrieved_docs_map or all_retrieved_scores_map[doc_key] < weighted_score:
                                all_retrieved_docs_map[doc_key] = doc
                                all_retrieved_scores_map[doc_key] = weighted_score
                except Exception as e:
                    logger.warning(f"Vector search failed for query '{q_str[:50]}...': {e}")
        
        
        initial_retrieved_list: List[Tuple[Document, float]] = sorted(
            [(doc, all_retrieved_scores_map[doc.page_content]) for doc in all_retrieved_docs_map.values()],
            key=lambda x: x[1], reverse=True
        )[:self.config.max_initial_retrieval_docs]
        
        final_docs_for_llm = [doc for doc, _ in initial_retrieved_list]
        relevance_scores = [score for _, score in initial_retrieved_list]

        
        if self.config.enable_cross_encoder_reranking and final_docs_for_llm:
            doc_contents = [doc.page_content for doc in final_docs_for_llm]
            reranked_tuples = self.embedding_manager.rerank_documents(
                query_context.original_query,
                doc_contents,
                self.config.max_reranked_docs
            )
            
            
            reranked_final_docs = []
            reranked_relevance_scores = []
            for idx, score in reranked_tuples:
                doc = final_docs_for_llm[idx]
                doc.metadata["relevance_score"] = score 
                reranked_final_docs.append(doc)
                reranked_relevance_scores.append(score)
            
            final_docs_for_llm = reranked_final_docs
            relevance_scores = reranked_relevance_scores
            
        
        if self.config.enable_document_graph_analysis and self.embedding_manager._parent_rag_pipeline is not None \
           and self.embedding_manager._parent_rag_pipeline.document_graph_builder.is_initialized \
           and query_context.entity_extraction:
            
            query_entity_texts = [ent['text'] for ent in query_context.entity_extraction if isinstance(ent, dict) and 'text' in ent]
            if query_entity_texts:
                graph_related_docs = self.embedding_manager._parent_rag_pipeline.document_graph_builder.find_related_documents_from_graph(
                    query_entity_texts, self.config.max_reranked_docs // 2 
                )
                
                
                for g_doc in graph_related_docs:
                    if g_doc.page_content not in {d.page_content for d in final_docs_for_llm}:
                        final_docs_for_llm.append(g_doc)
                        relevance_scores.append(0.3) 
                
                
                combined_docs_with_scores = sorted(
                    zip(final_docs_for_llm, relevance_scores),
                    key=lambda x: x[1], reverse=True
                )[:self.config.max_reranked_docs]
                final_docs_for_llm = [d for d, _ in combined_docs_with_scores]
                relevance_scores = [s for _, s in combined_docs_with_scores]
            
        processing_time = (datetime.now() - start_time).total_seconds()
        
        
        if relevance_scores:
            avg_relevance = np.mean(relevance_scores)
            num_docs_factor = min(len(final_docs_for_llm) / self.config.max_reranked_docs, 1.0)
            confidence_score = float(avg_relevance * 0.7 + num_docs_factor * 0.3) 
        else:
            confidence_score = 0.0

        return RetrievalResult(
            documents=final_docs_for_llm,
            relevance_scores=relevance_scores,
            confidence_score=confidence_score,
            retrieval_strategy="hybrid_multiquery_reranked_graph_aware",
            processing_time=processing_time,
            query_analysis=query_context.__dict__,
            document_clusters=list(self.document_clusters.values()), 
            cross_references=self.cross_references, 
            legal_citations=self._extract_legal_citations(final_docs_for_llm)
        )

    def _extract_legal_citations(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extracts legal citations from documents' metadata."""
        citations = []
        for doc in documents:
            citations.append({
                "source_file": doc.metadata.get("source_file", "unknown"),
                "page_number": doc.metadata.get("page_number", 1),
                "section_title": doc.metadata.get("section_title", ""),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "relevance_score": doc.metadata.get("relevance_score", 0.0)
            })
        return citations





class AdvancedRAGPipeline:
    """Main RAG pipeline orchestrating all components."""
    
    def __init__(self, 
                 cache_manager: Any, 
                 query_analyzer: Any, 
                 document_processor: AdvancedDocumentProcessor, 
                 model_orchestrator: Any, 
                 monitoring_engine: Any):
        
        self.cache_manager = cache_manager
        self.query_analyzer = query_analyzer
        self.document_processor = document_processor 
        self.model_orchestrator = model_orchestrator
        self.monitoring_engine = monitoring_engine
        
        self.config = RetrievalConfig()
        self.embedding_manager = AdvancedEmbeddingManager()
        
        self.embedding_manager._parent_rag_pipeline = self 

        self.document_graph_builder = DocumentGraphBuilder()
        
        self.document_graph_builder._parent_rag_pipeline = self 

        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.documents: List[Document] = [] 
        self.is_initialized = False
    
    async def initialize(self):
        """Initializes the entire RAG pipeline and its sub-components."""
        logger.info("Initializing Advanced RAG Pipeline...")
        try:
            
            await self.embedding_manager.initialize()
            await self.document_graph_builder.initialize() 

            
            await self._load_documents() 

            
            self.hybrid_retriever = HybridRetriever(self.embedding_manager, self.config)
            await self.hybrid_retriever.initialize(self.documents) 
            
            self.is_initialized = True
            logger.info("Advanced RAG Pipeline initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Advanced RAG Pipeline", error=str(e), exc_info=True)
            self.is_initialized = False 
            raise

    async def _load_documents(self):
        """Loads and processes legal documents from the specified directory using AdvancedDocumentProcessor."""
        legal_docs_path = Path("legal_docs")
        if not legal_docs_path.exists() or not any(legal_docs_path.iterdir()):
            logger.warning("`legal_docs` directory not found or is empty. RAG pipeline will have no knowledge base.")
            self.documents = [] 
            return

        logger.info(f"Scanning and processing documents from: {legal_docs_path.resolve()}")
        
        supported_extensions = ['.pdf', '.docx', '.txt', '.md', '.xlsx', '.zip', '.rar', '.7z']
        doc_paths = [p for p in legal_docs_path.rglob('*') if p.suffix.lower() in supported_extensions]

        if not doc_paths:
            logger.warning("No supported documents found in `legal_docs`. RAG pipeline will have no knowledge base.")
            self.documents = []
            return

        
        processed_docs_results: List[ProcessedDocument] = await self.document_processor.batch_process(
            [str(p) for p in doc_paths]
        )

        all_chunks: List[Document] = []
        for proc_doc in processed_docs_results:
            if proc_doc.metadata.status == ProcessingStatus.COMPLETED and proc_doc.chunks:
                for chunk in proc_doc.chunks:
                    
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
                            "entities": proc_doc.metadata.entities,
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
        
        
        if self.config.enable_document_graph_analysis and self.document_graph_builder.is_initialized and self.documents:
            self.document_graph_builder.build_document_graph(self.documents)


    async def enhanced_query(self, query: str, language: str, query_analysis: Dict[str, Any], query_id: str) -> Dict[str, Any]:
        """Processes an enhanced query through the RAG pipeline."""
        if not self.is_initialized or self.hybrid_retriever is None:
            raise RuntimeError("RAG Pipeline not initialized. Cannot process query.")
        
        start_time = datetime.now()
        
        
        
        entity_extraction_list_of_dicts = []
        for ent in query_analysis.get("entity_extraction", []):
            if isinstance(ent, str): 
                entity_extraction_list_of_dicts.append({'text': ent, 'label': 'UNKNOWN'})
            elif isinstance(ent, dict) and 'text' in ent and 'label' in ent:
                entity_extraction_list_of_dicts.append(ent)


        query_context_obj = QueryContext(
            original_query=query,
            expanded_queries=query_analysis.get("expanded_queries", []),
            intent_classification=query_analysis.get("intent_classification", {}),
            entity_extraction=entity_extraction_list_of_dicts, 
            legal_domain=query_analysis.get("legal_domain", ""),
            complexity_score=query_analysis.get("complexity_score", 0.0),
            urgency_level=query_analysis.get("urgency_level", "normal"),
            user_expertise=query_analysis.get("user_expertise", "general"),
            session_context=query_analysis.get("session_context", {}),
            temporal_context=query_analysis.get("temporal_context", {})
        )

        try:
            
            retrieval_result = await self.hybrid_retriever.retrieve(query_context_obj) 

            
            if self.config.enable_document_graph_analysis and self.document_graph_builder.is_initialized and query_context_obj.entity_extraction:
                query_entity_texts = [ent['text'] for ent in query_context_obj.entity_extraction if isinstance(ent, dict) and 'text' in ent]
                if query_entity_texts:
                    graph_related_docs = self.document_graph_builder.find_related_documents_from_graph(
                        query_entity_texts, self.config.max_reranked_docs // 2 
                    )
                    
                    
                    for g_doc in graph_related_docs:
                        if g_doc.page_content not in {d.page_content for d in retrieval_result.documents}:
                            retrieval_result.documents.append(g_doc)
                            
                            retrieval_result.relevance_scores.append(0.3) 
                    
                    
                    combined_docs_with_scores = sorted(
                        zip(retrieval_result.documents, retrieval_result.relevance_scores),
                        key=lambda x: x[1], reverse=True
                    )[:self.config.max_reranked_docs]
                    retrieval_result.documents = [d for d, _ in combined_docs_with_scores]
                    retrieval_result.relevance_scores = [s for _, s in combined_docs_with_scores]
                    retrieval_result.confidence_score = float(np.mean(retrieval_result.relevance_scores)) if retrieval_result.relevance_scores else 0.0


            
            response = await self.model_orchestrator.generate_response(
                query=query,
                context_documents=retrieval_result.documents,
                language=language,
                query_analysis=query_analysis
            )
            
            
            final_result = {
                "answer": response["answer"],
                "source_documents": retrieval_result.documents,
                "confidence_score": retrieval_result.confidence_score,
                "processing_time": retrieval_result.processing_time, 
                "retrieval_metadata": {
                    "strategy": retrieval_result.retrieval_strategy,
                    "legal_citations": retrieval_result.legal_citations,
                    "document_clusters": retrieval_result.document_clusters, 
                    "cross_references": retrieval_result.cross_references 
                }
            }
            
            await self.monitoring_engine.log_query_metrics({
                "query_id": query_id, 
                "processing_time": final_result["processing_time"],
                "documents_retrieved": len(retrieval_result.documents),
                "confidence_score": final_result["confidence_score"],
                "retrieval_strategy": final_result["retrieval_metadata"]["strategy"]
            })
            
            return final_result
            
        except Exception as e:
            logger.error("Error in enhanced query processing", query_id=query_id, error=str(e), exc_info=True)
            raise

    async def health_check(self) -> bool:
        """Checks the health of RAG pipeline components."""
        if not self.is_initialized:
            logger.warning("RAG Pipeline not initialized.")
            return False
        
        health_status = True
        
        
        if not self.embedding_manager.models or any(m is None for m in self.embedding_manager.models.values()):
            logger.error("Embedding manager models not fully loaded.")
            health_status = False
        
        
        if self.hybrid_retriever is None or not self.hybrid_retriever.documents_store:
            logger.error("Hybrid retriever not initialized or has no documents.")
            health_status = False

        
        if not self.document_graph_builder.is_initialized:
            logger.warning("Document graph builder not initialized. Graph features might be missing.")
        
        
        
        try:
            if hasattr(self.cache_manager, 'health_check') and not await self.cache_manager.health_check():
                logger.error("Cache manager failed health check.")
                health_status = False
        except Exception as e:
            logger.error(f"Error checking CacheManager health: {e}", exc_info=True)
            health_status = False

        try:
            if hasattr(self.query_analyzer, 'health_check') and not await self.query_analyzer.health_check():
                logger.error("Query analyzer failed health check.")
                health_status = False
        except Exception as e:
            logger.error(f"Error checking QueryAnalyzer health: {e}", exc_info=True)
            health_status = False

        try:
            
            if hasattr(self.document_processor, 'health_check') and not await self.document_processor.health_check():
                logger.error("Document processor failed health check.")
                health_status = False
        except Exception as e:
            logger.error(f"Error checking DocumentProcessor health: {e}", exc_info=True)
            health_status = False
        
        try:
            if hasattr(self.model_orchestrator, 'health_check') and not await self.model_orchestrator.health_check():
                logger.error("Model orchestrator failed health check.")
                health_status = False
        except Exception as e:
            logger.error(f"Error checking ModelOrchestrator health: {e}", exc_info=True)
            health_status = False

        try:
            if hasattr(self.monitoring_engine, 'health_check') and not await self.monitoring_engine.health_check():
                logger.error("Monitoring engine failed health check.")
                health_status = False
        except Exception as e:
            logger.error(f"Error checking MonitoringEngine health: {e}", exc_info=True)
            health_status = False

        return health_status






async def main_test():
    """Function to test the RAG pipeline in isolation with dummy components."""
    logger.info("--- Starting Standalone Advanced RAG Pipeline Test ---")
    
    
    cache = DummyCacheManager()
    analyzer = DummyQueryAnalyzer()
    
    
    doc_processor = AdvancedDocumentProcessor() 
    
    
    
    orchestrator = DummyModelOrchestrator()
    monitor = DummyMonitoringEngine()

    
    temp_legal_docs_dir = Path("legal_docs") 
    temp_legal_docs_dir.mkdir(exist_ok=True)
    
    sample_doc_path_1 = temp_legal_docs_dir / "penal_code_part1.txt"
    with open(sample_doc_path_1, "w") as f:
        f.write("ORGANIC LAW N° 01/2012/OL INSTITUTING THE PENAL CODE.\n")
        f.write("Article 1: Purpose of this Organic Law is to define criminal offences.\n")
        f.write("Article 2: An offence is any act punishable by law. This requires a fair trial and due process. My email is john.doe@example.com.")
    
    sample_doc_path_2 = temp_legal_docs_dir / "criminal_procedure_art.txt"
    with open(sample_doc_path_2, "w") as f:
        f.write("LAW N° 02/2012/OL ON CRIMINAL PROCEDURE.\n")
        f.write("Article 10: Investigation process must respect human rights. No punishment without law.\n")
        f.write("Article 11: The court shall apply penalties in accordance with national laws. Mr. Smith is a judge.")

    logger.info(f"Created sample documents in: {temp_legal_docs_dir}")

    
    pipeline = AdvancedRAGPipeline(cache, analyzer, doc_processor, orchestrator, monitor)
    await pipeline.initialize()

    
    is_healthy = await pipeline.health_check()
    logger.info(f"Pipeline health check after initialization: {'Healthy' if is_healthy else 'Unhealthy'}")
    assert is_healthy, "RAG Pipeline health check failed after initialization!"
    
    
    test_query = "What is an offence according to the penal code and what about investigations? Who is Mr. Smith?"
    logger.info(f"Running test query: '{test_query}'")
    
    query_analysis = await analyzer.comprehensive_analyze(test_query)
    result = await pipeline.enhanced_query(
        query=test_query,
        language="en",
        query_analysis=query_analysis,
        query_id="test-rag-1"
    )

    
    print("\n" + "="*80)
    print("                      ADVANCED RAG PIPELINE TEST RESULTS")
    print("="*80 + "\n")
    print(f"Original Query: {test_query}")
    print(f"Generated Answer: {result['answer']}")
    print(f"Overall Confidence: {result['confidence_score']:.2f}")
    print(f"Total Processing Time: {result['processing_time']:.2f}s")
    print("\n--- Retrieved Source Documents ---")
    if result['source_documents']:
        for i, doc in enumerate(result['source_documents']):
            print(f"  [{i+1}] Source: {doc.metadata.get('source_file', 'N/A')}, Page: {doc.metadata.get('page_number', 'N/A')}, Section: '{doc.metadata.get('section_title', 'N/A')}'")
            print(f"      Content Snippet: '{doc.page_content.strip()[:150]}...'")
            if doc.metadata.get("relevance_score") is not None:
                print(f"      Relevance Score: {doc.metadata['relevance_score']:.2f}")
    else:
        print("  No documents retrieved.")
    print("\n" + "="*80)
    
    
    await doc_processor.cleanup()
    
    if temp_legal_docs_dir.exists():
        try:
            shutil.rmtree(temp_legal_docs_dir) 
            logger.info(f"Successfully removed directory: {temp_legal_docs_dir}")
        except OSError as e:
            logger.error(f"Error removing directory {temp_legal_docs_dir}: {e}")
    logger.info("--- Standalone RAG Pipeline Test Finished and Cleaned Up ---")
    logger.info("--- Standalone RAG Pipeline Test Finished and Cleaned Up ---")


if __name__ == "__main__":
    asyncio.run(main_test())