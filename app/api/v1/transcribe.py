import asyncio

from fastapi import APIRouter, Depends, Form, UploadFile
from loguru import logger

from app.config import Settings
from app.core.gate import JobGate
from app.core.mlx_worker import MlxWorker
from app.dependencies import (
    get_gate,
    get_media_service,
    get_mlx_worker,
    get_settings,
    get_transcriber,
)
from app.schemas.transcription import TranscribeResponse, TranscribeUrlRequest
from app.services.media import MediaService, wav_duration_seconds
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
    gate: JobGate = Depends(get_gate),
    worker: MlxWorker = Depends(get_mlx_worker),
) -> TranscribeResponse:
    """Synchronous transcription — upload file, wait, receive transcript."""
    async with gate.reserve("transcribe") as job:
        await validate_upload(file, settings)
        input_path = await save_temp_file(file, settings)
        # Avoid in-place ffmpeg writes when upload is already .wav.
        wav_path = input_path.with_name(f"{input_path.stem}_extracted.wav")

        try:
            result = await _extract_and_transcribe(
                gate, worker, job, media, transcriber, input_path, wav_path, language
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
    gate: JobGate = Depends(get_gate),
    worker: MlxWorker = Depends(get_mlx_worker),
) -> TranscribeResponse:
    """Download a file from URL and transcribe it."""
    async with gate.reserve("transcribe") as job:
        input_path = await download_file_from_url(body.url, settings)
        wav_path = input_path.with_name(f"{input_path.stem}_extracted.wav")

        try:
            result = await _extract_and_transcribe(
                gate,
                worker,
                job,
                media,
                transcriber,
                input_path,
                wav_path,
                body.language,
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


async def _extract_and_transcribe(
    gate: JobGate,
    worker: MlxWorker,
    job,
    media: MediaService,
    transcriber: TranscriberService,
    input_path,
    wav_path,
    language: str | None,
) -> TranscribeResponse:
    """Run ffmpeg and mlx-whisper off the event loop, under the GPU gate.

    ffmpeg is CPU-only so it runs in any thread before the GPU slot is taken;
    the whisper call holds the slot and must run on the MLX worker thread.
    """
    await asyncio.to_thread(media.extract_audio, input_path, wav_path)
    audio_seconds = wav_duration_seconds(wav_path)
    async with gate.run(job, audio_seconds=audio_seconds):
        return await worker.run(
            transcriber.transcribe,
            str(wav_path),
            language=language if language != "auto" else None,
        )
