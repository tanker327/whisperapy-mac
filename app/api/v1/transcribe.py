from fastapi import APIRouter, Depends, Form, UploadFile
from loguru import logger

from app.config import Settings
from app.dependencies import get_media_service, get_settings, get_transcriber
from app.schemas.transcription import TranscribeResponse, TranscribeUrlRequest
from app.services.media import MediaService
from app.services.transcriber import TranscriberService
from app.utils.file_handler import (
    cleanup_temp,
    download_file_from_url,
    save_temp_file,
    validate_upload,
)

router = APIRouter(prefix="/transcribe", tags=["transcription"])


@router.post("", response_model=TranscribeResponse)
async def transcribe_sync(
    file: UploadFile,
    language: str = Form(default="auto"),
    include_segments: bool = Form(default=False),
    settings: Settings = Depends(get_settings),
    transcriber: TranscriberService = Depends(get_transcriber),
    media: MediaService = Depends(get_media_service),
) -> TranscribeResponse:
    """Synchronous transcription — upload file, wait, receive transcript."""
    await validate_upload(file, settings)
    input_path = await save_temp_file(file, settings)
    # Avoid in-place ffmpeg writes when upload is already .wav.
    wav_path = input_path.with_name(f"{input_path.stem}_extracted.wav")

    try:
        media.extract_audio(input_path, wav_path)
        result = transcriber.transcribe(
            str(wav_path),
            language=language if language != "auto" else None,
        )
        logger.info(
            f"Transcription complete: {file.filename} | "
            f"language={result.language_detected} | "
            f"duration={result.duration_seconds}s | "
            f"processing={result.processing_time_seconds}s"
        )
        if not include_segments:
            result.segments = []
        return result
    finally:
        cleanup_temp(input_path, wav_path)


@router.post("/url", response_model=TranscribeResponse)
async def transcribe_url(
    body: TranscribeUrlRequest,
    settings: Settings = Depends(get_settings),
    transcriber: TranscriberService = Depends(get_transcriber),
    media: MediaService = Depends(get_media_service),
) -> TranscribeResponse:
    """Download a file from URL and transcribe it."""
    input_path = await download_file_from_url(body.url, settings)
    wav_path = input_path.with_name(f"{input_path.stem}_extracted.wav")

    try:
        media.extract_audio(input_path, wav_path)
        result = transcriber.transcribe(
            str(wav_path),
            language=body.language if body.language != "auto" else None,
        )
        logger.info(
            f"Transcription complete: {body.url} | "
            f"language={result.language_detected} | "
            f"duration={result.duration_seconds}s | "
            f"processing={result.processing_time_seconds}s"
        )
        if not body.include_segments:
            result.segments = []
        return result
    finally:
        cleanup_temp(input_path, wav_path)
