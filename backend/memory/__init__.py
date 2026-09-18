"""
Memory management package for conversation history persistence.

Two backends are supported:
    - "memory"  →  InMemorySaver   (ephemeral, process-local, dev/test)
    - "sqlite"  →  SQLiteSaver     (persistent, file-based, DEFAULT)

The active backend is selected by MEMORY_BACKEND in .env.
"""
from backend.memory.base import BaseMemorySaver, MemoryBackend, SessionInfo, CheckpointRecord
from backend.memory.in_memory_saver import InMemorySaver
from backend.memory.sqlite_saver import SQLiteSaver

__all__ = [
    "BaseMemorySaver",
    "MemoryBackend",
    "SessionInfo",
    "CheckpointRecord",
    "InMemorySaver",
    "SQLiteSaver",
]
