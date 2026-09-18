"""
Pydantic models for pipeline job tracking.
"""
from pydantic import BaseModel
from typing import Literal, Optional


class PhaseStatus(BaseModel):
    name: str
    status: Literal["pending", "in_progress", "complete", "failed"]
    progress: int = 0  # 0-100
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "failed"]
    phases: dict  # phase_name -> PhaseStatus
    created_at: str
    updated_at: str
    error: Optional[str] = None
