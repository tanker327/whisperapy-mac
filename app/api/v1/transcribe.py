import json

from fastapi import APIRouter, Depends, Form, UploadFile, WebSocket, WebSocketDisconnect
from loguru import logger

from app.config import Settings
from app.dependencies import get_media_service, get_settings, get_transcriber
from app.schemas.transcription import TranscribeResponse, TranscribeUrlRequest
from app.services.media import MediaService
from app.services.streaming import StreamingSession
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


async def _emit(
    websocket: WebSocket, committed: list[dict], partial: dict | None
) -> None:
    if committed:
        await websocket.send_json({"type": "final", "segments": committed})
    if partial:
        await websocket.send_json({"type": "partial", "segment": partial})


@router.websocket("/stream")
async def transcribe_stream(websocket: WebSocket) -> None:
    """Realtime streaming transcription over a WebSocket.

    Protocol:
      - (optional) first send a JSON text frame ``{"type": "config",
        "language": "en"}`` to override language detection.
      - Send binary frames of raw 16-bit signed little-endian PCM, mono, at the
        server's ``stream_sample_rate`` (default 16kHz).
      - Send ``{"type": "flush"}`` to force a transcription pass, or
        ``{"type": "end"}`` to finalize and receive the aggregate result.

    Server emits JSON text frames: ``partial`` (still-forming tail segment),
    ``final`` (stabilized segments), ``done`` (aggregate), and ``error``.
    """
    await websocket.accept()
    settings = get_settings()
    transcriber = get_transcriber()

    if not transcriber.is_ready():
        await websocket.send_json({"type": "error", "message": "Model not ready"})
        await websocket.close()
        return

    session = StreamingSession(
        transcriber, settings, language=settings.default_language
    )
    logger.info("Streaming session started")

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if (data := message.get("bytes")) is not None:
                session.add_pcm(data)
                if session.should_process():
                    await _emit(websocket, *await session.process())
                continue

            text = message.get("text")
            if text is None:
                continue
            try:
                control = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON control message"}
                )
                continue

            mtype = control.get("type")
            if mtype == "config":
                session.set_language(control.get("language"))
            elif mtype == "flush":
                await _emit(websocket, *await session.process())
            elif mtype == "end":
                await _emit(websocket, *await session.process(final=True))
                await websocket.send_json(session.summary())
                break
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown control type: {mtype!r}"}
                )
    except WebSocketDisconnect:
        logger.info("Streaming client disconnected")
    except Exception as e:  # noqa: BLE001 - surface to client, never crash the server
        logger.exception("Streaming session error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Streaming session ended")
