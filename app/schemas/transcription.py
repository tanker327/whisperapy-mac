from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class OutputFormat(StrEnum):
    json = "json"
    verbose_json = "verbose_json"
    text = "text"
    srt = "srt"
    vtt = "vtt"


class TranscribeParams(BaseModel):
    """Options shared by the upload, URL, and OpenAI-compatible endpoints."""

    language: str = "auto"
    include_segments: bool = False
    word_timestamps: bool = False
    initial_prompt: str | None = Field(default=None, max_length=2000)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    condition_on_previous_text: bool = True
    output_format: OutputFormat = OutputFormat.json


class TranscribeUrlRequest(TranscribeParams):
    url: HttpUrl


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class TranscribeResponse(BaseModel):
    job_id: str
    language_detected: str | None = None
    duration_seconds: float | None = None
    processing_time_seconds: float | None = None
    text: str = ""
    segments: list[Segment] = Field(default_factory=list)
