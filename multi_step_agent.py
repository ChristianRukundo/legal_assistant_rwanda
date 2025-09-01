"""
MultiStepQueryAgent: Orchestrates the process of handling complex queries by
planning, executing sub-queries, and synthesizing a final response.
Version: 1.0.1 (Type-Safe Hotfix)
"""

import asyncio
import logging
from typing import List, Dict, Any, Union

import structlog

from rag_pipeline import RAGPipeline
from conversation_context_service import ConversationContextService
from query_processor import QueryProcessor, ProcessedQuery
from production_components import ProductionModelOrchestrator
from query_planner import QueryPlanner, DecompositionPlan

logger = structlog.get_logger(__name__)


class MultiStepQueryAgent:
    """An agent that can plan and execute multi-step RAG queries."""

    def __init__(
        self,
        query_planner: QueryPlanner,
        query_analyzer: QueryProcessor,
        rag_pipeline: RAGPipeline,
        model_orchestrator: ProductionModelOrchestrator,
    ):
        self.query_planner = query_planner
        self.query_analyzer = query_analyzer
        self.rag_pipeline = rag_pipeline
        self.model_orchestrator = model_orchestrator

    async def execute(
        self,
        query: str,
        language: str,
        session_id: str,
        query_id: str,
        conversation_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Executes the full agentic process for a given query.
        """
        logger.info(
            "Multi-step agent received new query.",
            query=query,
            query_id=query_id,
            session_id=session_id,
        )

        plan = await self.query_planner.create_plan(query)

        if len(plan.plan) <= 1:
            logger.info("Executing as a single-step query.", query_id=query_id)
            return await self.rag_pipeline.enhanced_query(
                query=query,
                language=language,
                query_analysis=await self.query_analyzer.comprehensive_analyze(
                    query=query,
                    language=language,
                    session_id=session_id,
                    conversation_history=conversation_history,
                ),
                query_id=query_id,
                conversation_history=conversation_history,
            )

        logger.info(
            f"Executing multi-step plan with {len(plan.plan)} steps.", query_id=query_id
        )

        sub_query_tasks = []
        for sub_query in plan.plan:
            task = self._execute_sub_query(
                sub_query=sub_query.question,
                language=language,
                session_id=session_id,
                query_id=query_id,
                conversation_history=conversation_history,
            )
            sub_query_tasks.append(task)

        gathered_results: List[Union[Dict[str, Any], BaseException]] = (
            await asyncio.gather(*sub_query_tasks, return_exceptions=True)
        )

        processed_intermediate_results: List[Dict[str, Any]] = []
        for i, result in enumerate(gathered_results):
            sub_q_text = plan.plan[i].question

            if isinstance(result, BaseException):
                logger.error(
                    "Sub-query execution failed with an exception.",
                    sub_query=sub_q_text,
                    error=str(result),
                    query_id=query_id,
                )

                failure_result = {
                    "question": sub_q_text,
                    "answer": f"I encountered an error trying to answer this part of the question.",
                    "citations": [],
                }
                processed_intermediate_results.append(failure_result)

            elif isinstance(result, dict):

                processed_intermediate_results.append(result)

            else:
                logger.error(
                    "Sub-query returned an unexpected type.",
                    sub_query=sub_q_text,
                    type=type(result),
                    query_id=query_id,
                )

                unknown_type_result = {
                    "question": sub_q_text,
                    "answer": "An unknown error occurred while processing this part of the question.",
                    "citations": [],
                }
                processed_intermediate_results.append(unknown_type_result)

        return await self._synthesize_final_response(
            plan, processed_intermediate_results
        )

    async def _execute_sub_query(
        self,
        sub_query: str,
        language: str,
        session_id: str,
        query_id: str,
        conversation_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Helper to execute a single step of the plan. This method is designed to
        propagate exceptions up to the `gather` call.
        """
        query_analysis = await self.query_analyzer.comprehensive_analyze(
            query=sub_query,
            language=language,
            session_id=session_id,
            conversation_history=conversation_history,
        )

        sub_query_log_id = (
            f"{query_id}-{query_analysis.intent.value}-{len(conversation_history)+1}"
        )

        result = await self.rag_pipeline.enhanced_query(
            query=sub_query,
            language=language,
            query_analysis=query_analysis,
            query_id=sub_query_log_id,
            conversation_history=conversation_history,
        )

        return {
            "question": sub_query,
            "answer": result["answer"],
            "citations": result["retrieval_metadata"]["legal_citations"],
        }

    async def _synthesize_final_response(
        self, plan: DecompositionPlan, intermediate_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combines answers from sub-queries into a single, coherent response."""

        context_str = ""
        for i, result in enumerate(intermediate_results):

            context_str += f"--- Answer to Sub-Question {i+1}: '{result.get('question', 'N/A')}' ---\n"
            context_str += f"{result.get('answer', 'No answer was generated.')}\n\n"

        prompt = f"""<s>[INST] You are an expert response synthesizer. Your task is to combine the provided "INDIVIDUAL ANSWERS" into a single, comprehensive, and well-structured answer to the "ORIGINAL USER QUERY".
- Use markdown for clear formatting (e.g., headings for each sub-question, bullet points).
- Do not add any new information. Synthesize ONLY from the provided answers.
- If an answer indicates a failure or error, state that clearly for that part of the question.
- Ensure the final answer is coherent, easy to read, and directly addresses all parts of the user's original query.

ORIGINAL USER QUERY: {plan.original_query}

INDIVIDUAL ANSWERS:
{context_str}

Synthesized Answer:
[/INST]"""

        final_answer = await self.model_orchestrator.generate_raw(
            prompt, max_new_tokens=1024
        )

        all_citations = [
            citation
            for result in intermediate_results
            for citation in result.get("citations", [])
        ]

        unique_citations = list(
            {frozenset(d.items()): d for d in all_citations}.values()
        )

        return {
            "answer": final_answer,
            "source_documents": [],
            "confidence_score": 0.95,
            "processing_time": 0,
            "retrieval_metadata": {
                "strategy": "multi_step_agentic_rag",
                "legal_citations": unique_citations,
            },
        }
