from fastapi import APIRouter, Depends, Form, UploadFile
from loguru import logger

from app.config import Settings
from app.dependencies import get_media_service, get_settings, get_transcriber
from app.schemas.transcription import TranscribeResponse
from app.services.media import MediaService
from app.services.transcriber import TranscriberService
from app.utils.file_handler import cleanup_temp, save_temp_file, validate_upload

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
    wav_path = input_path.with_suffix(".wav")

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
