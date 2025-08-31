"""
Data Models and Database Initialization for the Inyandiko Knowledge Base.
Uses aiosqlite for asynchronous database operations with SQLite.
"""

import asyncio
import logging
from pathlib import Path
import aiosqlite

logger = logging.getLogger(__name__)

DB_FILE = Path("vector_db") / "knowledge_base.db"


async def get_db_connection() -> aiosqlite.Connection:
    """Creates and returns an async database connection."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(DB_FILE)
    db.row_factory = aiosqlite.Row
    return db


async def initialize_database():
    """
    Initializes the database by creating the necessary tables if they don't exist.
    This function is idempotent.
    """
    logger.info(f"Initializing knowledge base database at {DB_FILE}...")
    try:
        async with await get_db_connection() as db:
            # Table for storing document metadata
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Table for storing document chunks
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    chunk_index INTEGER NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Table for storing individual turns within a session
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    user_query TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """
            )

            # Indexes for faster lookups
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents (file_path)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents (file_hash)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)"
            )

            await db.commit()
            logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(initialize_database())
    print("Database schema created successfully (if it did not exist).")
