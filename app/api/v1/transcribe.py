import uuid

from fastapi import APIRouter, Depends, Form, UploadFile
from loguru import logger

from app.config import Settings
from app.dependencies import get_media_service, get_settings, get_transcriber
from app.schemas.transcription import (
    JobStatus,
    JobStatusResponse,
    OutputFormat,
    TranscribeResponse,
)
from app.services.media import MediaService
from app.services.transcriber import TranscriberService
from app.utils.file_handler import cleanup_temp, save_temp_file, validate_upload

router = APIRouter(prefix="/transcribe", tags=["transcription"])

# In-memory job store for async jobs
_jobs: dict[str, JobStatusResponse] = {}


@router.post("", response_model=TranscribeResponse)
async def transcribe_sync(
    file: UploadFile,
    language: str = Form(default="auto"),
    word_timestamps: bool = Form(default=False),
    output_format: OutputFormat = Form(default=OutputFormat.json),
    settings: Settings = Depends(get_settings),
    transcriber: TranscriberService = Depends(get_transcriber),
    media: MediaService = Depends(get_media_service),
) -> TranscribeResponse:
    """Synchronous transcription — upload file, wait, receive transcript."""
    await validate_upload(file, settings)
    input_path = await save_temp_file(file, settings)
    wav_path = input_path.with_suffix(".wav")

    try:
        media.extract_audio(input_path, wav_path)
        result = transcriber.transcribe(
            str(wav_path),
            language=language if language != "auto" else None,
            word_timestamps=word_timestamps,
        )
        logger.info(
            f"Transcription complete: {file.filename} | "
            f"language={result.language_detected} | "
            f"duration={result.duration_seconds}s | "
            f"processing={result.processing_time_seconds}s"
        )
        return result
    finally:
        cleanup_temp(input_path, wav_path)


@router.post("/jobs", response_model=JobStatusResponse)
async def transcribe_async(
    file: UploadFile,
    language: str = Form(default="auto"),
    word_timestamps: bool = Form(default=False),
    output_format: OutputFormat = Form(default=OutputFormat.json),
    settings: Settings = Depends(get_settings),
) -> JobStatusResponse:
    """Async transcription — upload file, receive job_id immediately."""
    await validate_upload(file, settings)
    await save_temp_file(file, settings)

    job_id = str(uuid.uuid4())
    job = JobStatusResponse(job_id=job_id, status=JobStatus.pending)
    _jobs[job_id] = job

    # Store job metadata for background processing
    # NOTE: In v1, we just mark it as pending. A background worker would pick it up.
    logger.info(f"Job created: {job_id} for {file.filename}")
    return job


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll job status and retrieve result."""
    job = _jobs.get(job_id)
    if job is None:
        from app.core.exceptions import WhisperapyError

        raise WhisperapyError(f"Job {job_id} not found")
    return job


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    """Cancel job and clean up."""
    job = _jobs.pop(job_id, None)
    if job is None:
        from app.core.exceptions import WhisperapyError

        raise WhisperapyError(f"Job {job_id} not found")
    return {"job_id": job_id, "status": "cancelled"}
