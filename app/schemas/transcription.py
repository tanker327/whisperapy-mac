from enum import Enum

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    json = "json"
    text = "text"
    srt = "srt"
    vtt = "vtt"


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


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


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float | None = None
    error: str | None = None
    result: TranscribeResponse | None = None
