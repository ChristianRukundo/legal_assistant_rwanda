"""
ConversationContextService: Manages the storage and retrieval of conversation
histories from the database, enabling session-based memory for the assistant.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from data_models import get_db_connection

logger = logging.getLogger(__name__)


class ConversationContextService:
    """Handles all database interactions for conversation sessions and turns."""

    async def get_or_create_session(
        self, session_id: str, user_id: Optional[str] = None
    ) -> None:
        """
        Ensures a session exists in the database. If not, it creates one.
        Also updates the last_accessed timestamp for the session.
        """
        async with await get_db_connection() as db:
            async with db.executescript("BEGIN TRANSACTION;"):
                cursor = await db.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                session = await cursor.fetchone()

                if session:
                    # Update last accessed time
                    await db.execute(
                        "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                        (datetime.utcnow(), session_id),
                    )
                else:
                    # Create a new session
                    await db.execute(
                        "INSERT INTO sessions (session_id, user_id, last_accessed) VALUES (?, ?, ?)",
                        (session_id, user_id, datetime.utcnow()),
                    )
                await db.commit()

    async def add_turn(
        self, session_id: str, user_query: str, assistant_response: str
    ) -> None:
        """Adds a new user-assistant turn to the conversation history."""
        async with await get_db_connection() as db:
            async with db.executescript("BEGIN TRANSACTION;"):
                # Get the last turn index to determine the new one
                cursor = await db.execute(
                    "SELECT MAX(turn_index) as max_index FROM turns WHERE session_id = ?",
                    (session_id,),
                )
                last_turn = await cursor.fetchone()
                new_index = (
                    (last_turn["max_index"] + 1)
                    if last_turn and last_turn["max_index"] is not None
                    else 1
                )

                # Insert the new turn
                await db.execute(
                    """INSERT INTO turns (session_id, turn_index, user_query, assistant_response)
                       VALUES (?, ?, ?, ?)""",
                    (session_id, new_index, user_query, assistant_response),
                )
                await db.commit()
        logger.info(f"Added turn {new_index} to session {session_id}")

    async def get_history(
        self, session_id: str, limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Retrieves the most recent turns from a conversation history.

        Returns:
            A list of dictionaries, where each dict is a turn with 'user' and 'assistant' keys.
        """
        history = []
        async with await get_db_connection() as db:
            cursor = await db.execute(
                """SELECT user_query, assistant_response FROM turns
                   WHERE session_id = ? ORDER BY turn_index DESC LIMIT ?""",
                (session_id, limit),
            )
            rows = await cursor.fetchall()
            # Reverse the list to get chronological order (oldest first)
            for row in reversed(list(rows)):
                history.append(
                    {"user": row["user_query"], "assistant": row["assistant_response"]}
                )
        return history

    def format_history_for_prompt(self, history: List[Dict[str, str]]) -> str:
        """
        Formats the conversation history into a simple, readable string for an LLM prompt.
        """
        if not history:
            return "No previous conversation history."

        formatted_string = ""
        for turn in history:
            formatted_string += (
                f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            )

        return formatted_string.strip()
