"""
QueryPlanner: A service that decomposes a complex user query into a structured,
executable plan of simpler sub-queries using an LLM.
"""

import logging
import orjson
from typing import List

from pydantic import BaseModel, Field, ValidationError
import structlog

from production_components import ProductionModelOrchestrator


logger = structlog.get_logger(__name__)


class SubQuery(BaseModel):
    """Represents a single, solvable question in the execution plan."""

    sub_query_id: int = Field(
        ..., description="A unique identifier for the sub-query, e.g., 1, 2, 3."
    )
    question: str = Field(
        ..., description="The simple, self-contained question to be answered."
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="List of sub_query_ids this query depends on (for future use).",
    )


class DecompositionPlan(BaseModel):
    """Represents the full, structured plan for answering a complex query."""

    original_query: str
    plan: List[SubQuery]


class QueryPlanner:
    """Uses an LLM to decompose a complex query into a structured plan."""

    def __init__(self, model_orchestrator: ProductionModelOrchestrator):
        self.model_orchestrator = model_orchestrator

    async def create_plan(self, complex_query: str) -> DecompositionPlan:
        """
        Creates an execution plan by decomposing a complex query.

        If decomposition fails or the query is simple, it returns a single-step plan.
        """
        prompt = f"""<s>[INST] You are an expert query analyst. Your task is to decompose a potentially complex user query into a series of simple, self-contained questions.
- If the query is already simple and asks only one thing, return a plan with a single sub-query.
- Identify distinct topics or questions within the query.
- Your response MUST be a single, valid JSON object that strictly follows this Pydantic model:
{{
  "original_query": "The user's original, unmodified query",
  "plan": [
    {{
      "sub_query_id": 1,
      "question": "The first simple question",
      "dependencies": []
    }},
    {{
      "sub_query_id": 2,
      "question": "The second simple question",
      "dependencies": []
    }}
  ]
}}

USER QUERY: {complex_query}
[/INST]"""

        try:
            response_text = await self.model_orchestrator.generate_raw(
                prompt, max_new_tokens=300
            )

            json_match = response_text[
                response_text.find("{") : response_text.rfind("}") + 1
            ]

            if json_match:
                plan_dict = orjson.loads(json_match)
                plan = DecompositionPlan(**plan_dict)

                logger.info("Successfully created multi-step plan.", plan=plan.dict())
                return plan

            raise ValueError("LLM did not return a parsable JSON object.")

        except (orjson.JSONDecodeError, ValidationError, ValueError) as e:

            logger.warning(
                "Failed to create or validate a multi-step plan. Defaulting to a single-step plan.",
                query=complex_query,
                error=str(e),
            )

            return DecompositionPlan(
                original_query=complex_query,
                plan=[SubQuery(sub_query_id=1, question=complex_query)],
            )
