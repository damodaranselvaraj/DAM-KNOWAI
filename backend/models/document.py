"""
Pydantic models for document management.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class DocumentMetadata(BaseModel):
    doc_id: str
    name: str
    file_type: str
    size_bytes: int
    pages: int
    status: Literal["indexed", "processing", "failed", "deleted"]
    uploaded_at: str
    chunk_count: int = 0


class DocumentStats(BaseModel):
    total_documents: int
    total_chunks: int
    total_size_bytes: int
    by_type: dict
    last_updated: str


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    size_bytes: int
    status: str
    message: str
