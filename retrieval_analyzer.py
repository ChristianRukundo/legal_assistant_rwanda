"""
RetrievalAnalyzer: A service that uses the LLM for meta-analysis of the retrieval process.
It can judge the relevance of retrieved context and reformulate queries to improve results.
"""

import logging
import orjson
from typing import List

from langchain.schema import Document
from production_components import ProductionModelOrchestrator

logger = logging.getLogger(__name__)


class RetrievalAnalyzer:
    """Uses an LLM to judge and improve retrieval results."""

    def __init__(self, model_orchestrator: ProductionModelOrchestrator):
        """
        Initializes the analyzer with a model orchestrator for making LLM calls.

        Args:
            model_orchestrator: An instance of ProductionModelOrchestrator.
        """
        self.model_orchestrator = model_orchestrator

    async def is_context_relevant(
        self, query: str, context_docs: List[Document]
    ) -> bool:
        """
        Judges whether the provided context documents are relevant enough to answer the query.

        Args:
            query: The user's original query.
            context_docs: The list of documents retrieved from the initial search.

        Returns:
            True if the context is deemed relevant, False otherwise.
        """
        if not context_docs:
            return False

        context_str = "\n---\n".join(
            [
                f"Document {i+1}: {doc.page_content[:500]}..."
                for i, doc in enumerate(context_docs)
            ]
        )

        prompt = f"""<s>[INST] You are a highly analytical AI judge. Your task is to determine if the provided CONTEXT contains sufficient information to directly answer the USER QUERY.
Focus only on relevance, not on the quality of the answer.
Respond with only the single word 'Yes' or 'No'.

CONTEXT:
{context_str}

USER QUERY: {query}

Is the context sufficient to answer the query? Answer with only 'Yes' or 'No'. [/INST]"""

        try:
            # Using a very short max_tokens for a simple classification task
            response = await self.model_orchestrator.generate_raw(
                prompt, max_new_tokens=5
            )
            # Be liberal in what we accept as a "Yes"
            if "yes" in response.lower():
                logger.info("Retrieval Judge: Context is RELEVANT.")
                return True
            else:
                logger.warning("Retrieval Judge: Context is NOT RELEVANT.")
                return False
        except Exception as e:
            logger.error(
                f"Error in Retrieval Judge LLM call: {e}. Defaulting to relevant to avoid breaking flow."
            )
            # Default to True to prevent the pipeline from breaking if the judge fails
            return True

    async def reformulate_query(self, query: str) -> List[str]:
        """
        Generates alternative, semantically similar queries to improve retrieval results.

        Args:
            query: The original user query that failed to retrieve good results.

        Returns:
            A list of new query strings to try.
        """
        prompt = f"""<s>[INST] You are an expert search query reformulator. The user's original query failed to find good results.
Your task is to generate 3 alternative search queries. The queries should be:
1.  Semantically similar to the original.
2.  Diverse in wording and keywords.
3.  Phrased as direct questions or searches.

Return your answer as a single, valid JSON list of strings. For example: ["query 1", "query 2", "query 3"]

ORIGINAL QUERY: {query} [/INST]"""

        try:
            response = await self.model_orchestrator.generate_raw(
                prompt, max_new_tokens=150
            )

            # Extract JSON from the potentially messy LLM output
            json_match = response[response.find("[") : response.rfind("]") + 1]

            if json_match:
                queries = orjson.loads(json_match)
                if isinstance(queries, list) and all(
                    isinstance(q, str) for q in queries
                ):
                    logger.info(f"Query Reformulator generated new queries: {queries}")
                    return queries

            logger.warning(
                "Query Reformulator failed to produce valid JSON. No new queries will be used."
            )
            return []
        except Exception as e:
            logger.error(
                f"Error in Query Reformulator LLM call: {e}. Returning no new queries."
            )
            return []
