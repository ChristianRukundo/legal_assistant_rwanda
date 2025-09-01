"""
Data models and database connection management for the application.
"""

import asyncio
from pathlib import Path
import aiosqlite
import structlog

logger = structlog.get_logger(__name__)

DB_FILE = Path("vector_db/knowledge_base.db")


def get_db_connection() -> aiosqlite.Connection:
    """
    Returns a new connection context to the SQLite database.
    This should be used in an 'async with' block.
    Example:
        async with get_db_connection() as db:
            await db.execute(...)
    """
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    return aiosqlite.connect(DB_FILE)


async def initialize_database():
    """
    Initializes the database by creating necessary tables if they don't exist.
    """
    logger.info(f"Initializing knowledge base database at {DB_FILE}...")
    try:

        async with get_db_connection() as db:

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    processed_at TIMESTAMP
                )
                """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    chunk_id INTEGER,
                    FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
                )
                """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    user_query TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, turn_index)
                )
                """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            await db.commit()
            logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise
