"""
DirectoryWatcherService: Monitors the legal documents directory for changes and
triggers the DocumentIngestionService to keep the knowledge base up-to-date.
"""
import asyncio
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from document_ingestion_service import DocumentIngestionService

logger = logging.getLogger(__name__)

class IngestionEventHandler(FileSystemEventHandler):
    """Handles file system events and queues them for processing."""
    def __init__(self, ingestion_service: DocumentIngestionService, loop: asyncio.AbstractEventLoop):
        self.ingestion_service = ingestion_service
        self.loop = loop
        self.supported_extensions = {'.pdf', '.docx', '.txt', '.md'}

    def _schedule_task(self, task_coro):
        """Schedules a coroutine to run on the main event loop."""
        asyncio.run_coroutine_threadsafe(task_coro, self.loop)

    def _is_supported(self, path: str) -> bool:
        """Check if the file has a supported extension."""
        return Path(path).suffix.lower() in self.supported_extensions

    def _get_str_path(self, src_path):
        if isinstance(src_path, bytes):
            return src_path.decode()
        elif isinstance(src_path, (bytearray, memoryview)):
            return bytes(src_path).decode()
        return str(src_path)

    def on_created(self, event: FileSystemEvent):
        src_path = self._get_str_path(event.src_path)
        if not event.is_directory and self._is_supported(src_path):
            logger.info(f"File created: {src_path}. Scheduling for ingestion.")
            self._schedule_task(self.ingestion_service.process_and_index_document(Path(src_path)))

    def on_modified(self, event: FileSystemEvent):
        src_path = self._get_str_path(event.src_path)
        if not event.is_directory and self._is_supported(src_path):
            logger.info(f"File modified: {src_path}. Scheduling for re-ingestion.")
            self._schedule_task(self.ingestion_service.process_and_index_document(Path(src_path)))

    def on_deleted(self, event: FileSystemEvent):
        src_path = self._get_str_path(event.src_path)
        if not event.is_directory and self._is_supported(src_path):
            logger.info(f"File deleted: {src_path}. Scheduling for deletion from knowledge base.")
            self._schedule_task(self.ingestion_service.delete_document(src_path))


class DirectoryWatcherService:
    """Manages the lifecycle of the directory observer."""
    def __init__(self, watch_path: Path, ingestion_service: DocumentIngestionService):
        self.watch_path = watch_path
        self.ingestion_service = ingestion_service
        self.observer = Observer()

    def start(self):
        """Starts the file system observer in a separate thread."""
        self.watch_path.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        event_handler = IngestionEventHandler(self.ingestion_service, loop)
        self.observer.schedule(event_handler, str(self.watch_path), recursive=True)
        self.observer.start()
        logger.info(f"Directory watcher started on: {self.watch_path}")

    def stop(self):
        """Stops the file system observer."""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            logger.info("Directory watcher stopped.")