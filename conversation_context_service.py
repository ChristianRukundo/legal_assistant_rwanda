"""
Manages conversational context and history in the database.
"""

import aiosqlite
from datetime import datetime
from typing import List, Dict

from data_models import get_db_connection


class ConversationContextService:
    """Manages conversational context and history in the database."""

    def __init__(self):
        """Initializes the context service."""
        pass

    async def get_or_create_session(self, session_id: str) -> None:
        """
        Ensures a session exists in the database.
        If the session does not exist, it will be created.
        """
        # The line below was `async with await get_db_connection() as db:`,
        # which caused a `RuntimeError: threads can only be started once`.
        # The `await` is incorrect because `aiosqlite.connect()` (which is
        # what get_db_connection wraps) returns a coroutine that also acts
        # as an async context manager. The `async with` statement handles
        # awaiting it correctly.
        async with get_db_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            )
            session = await cursor.fetchone()
            if not session:
                await db.execute(
                    "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
                    (session_id, datetime.utcnow()),
                )
                await db.commit()

    async def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves conversation history for a given session."""
        async with get_db_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT user_query, assistant_response FROM conversation_turns WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            )
            rows = await cursor.fetchall()
            return [
                {"user": row["user_query"], "assistant": row["assistant_response"]}
                for row in rows
            ]

    async def add_turn(
        self, session_id: str, user_query: str, assistant_response: str
    ) -> None:
        """Adds a new turn to the conversation history."""
        async with get_db_connection() as db:
            await db.execute(
                "INSERT INTO conversation_turns (session_id, user_query, assistant_response, created_at) VALUES (?, ?, ?, ?)",
                (session_id, user_query, assistant_response, datetime.utcnow()),
            )
            await db.commit()

    def format_history_for_prompt(self, history: List[Dict[str, str]]) -> str:
        """Formats conversation history into a string for LLM prompts."""
        if not history:
            return "No previous conversation history."

        formatted_history = "\n".join(
            [
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                for turn in history
            ]
        )
        return formatted_history
