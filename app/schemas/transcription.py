from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TranscribeUrlRequest(BaseModel):
    url: str
    language: str = "auto"
    include_segments: bool = False


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.completed
    language_detected: str | None = None
    duration_seconds: float | None = None
    processing_time_seconds: float | None = None
    text: str = ""
    segments: list[Segment] = Field(default_factory=list)
