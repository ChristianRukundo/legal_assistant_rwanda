"""
Database models and initialization for the Inyandiko Legal AI Assistant.
"""

import aiosqlite
import structlog

DB_PATH = "inyandiko_kb.db"
logger = structlog.get_logger(__name__)


def get_db_connection():
    """Returns a coroutine that creates an aiosqlite connection."""
    return aiosqlite.connect(DB_PATH)


async def initialize_database():
    """
    Initializes the SQLite database, creating all necessary tables if they don't exist.
    """
    logger.info(f"Initializing database at {DB_PATH}...")
    async with get_db_connection() as db:
        # Table for ingested documents
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            );
        """
        )

        # Table for document chunks
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                page_number INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
            );
        """
        )

        # Table for conversation sessions
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL
            );
        """
        )

        # Table for individual conversation turns (user query + assistant response)
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_query TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            );
        """
        )

        await db.commit()
        logger.info("Database schema initialized successfully.")
