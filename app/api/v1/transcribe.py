"""Transcription endpoints.

Three routes share one pipeline (``_transcribe_source``):

* ``POST /api/v1/transcribe``            multipart upload, native parameters
* ``POST /api/v1/transcribe/url``        JSON body with a URL
* ``POST /api/v1/audio/transcriptions``  OpenAI-compatible multipart shape

Pipeline: reserve a gate slot -> obtain the input file -> ffmpeg to WAV in a
thread -> load WAV to numpy -> take the GPU slot -> whisper on the MLX thread
-> render in the requested output format -> delete temp files.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Form, UploadFile
from fastapi.responses import PlainTextResponse, Response
from loguru import logger

from app.core.gate import Job, JobGate
from app.core.mlx_worker import MlxWorker
from app.dependencies import (
    GateDep,
    MediaDep,
    SettingsDep,
    TranscriberDep,
    WorkerDep,
)
from app.schemas.transcription import (
    OutputFormat,
    TranscribeParams,
    TranscribeResponse,
    TranscribeUrlRequest,
)
from app.services.media import MediaService
from app.services.transcriber import TranscribeOptions, TranscriberService
from app.utils import formats
from app.utils.file_handler import (
    cleanup_temp,
    download_file_from_url,
    save_temp_file,
    validate_upload,
)

router = APIRouter(tags=["transcription"])

_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {
            "application/json": {},
            "text/plain": {},
            "application/x-subrip": {},
            "text/vtt": {},
        }
    }
}

# Note on uploads: FastAPI has already received the multipart body before the
# handler runs, so a busy rejection here still costs the client its upload
# bandwidth. Clients that care can poll /health first. The URL endpoint
# genuinely rejects before any download starts.


@router.post("/transcribe", response_model=TranscribeResponse, responses=_RESPONSES)
async def transcribe_upload(
    file: UploadFile,
    settings: SettingsDep,
    transcriber: TranscriberDep,
    media: MediaDep,
    gate: GateDep,
    worker: WorkerDep,
    language: Annotated[str, Form()] = "auto",
    include_segments: Annotated[bool, Form()] = False,
    word_timestamps: Annotated[bool, Form()] = False,
    initial_prompt: Annotated[str | None, Form(max_length=2000)] = None,
    temperature: Annotated[float | None, Form(ge=0.0, le=1.0)] = None,
    condition_on_previous_text: Annotated[bool, Form()] = True,
    output_format: Annotated[OutputFormat, Form()] = OutputFormat.json,
) -> Response:
    """Upload a media file and receive its transcript."""
    params = TranscribeParams(
        language=language,
        include_segments=include_segments,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        temperature=temperature,
        condition_on_previous_text=condition_on_previous_text,
        output_format=output_format,
    )

    async def obtain() -> Path:
        await validate_upload(file, settings)
        return await save_temp_file(file, settings)

    return await _transcribe_source(
        obtain, file.filename or "upload", params, media, transcriber, gate, worker
    )


@router.post("/transcribe/url", response_model=TranscribeResponse, responses=_RESPONSES)
async def transcribe_url(
    body: TranscribeUrlRequest,
    settings: SettingsDep,
    transcriber: TranscriberDep,
    media: MediaDep,
    gate: GateDep,
    worker: WorkerDep,
) -> Response:
    """Download a media file from a public URL and transcribe it."""

    async def obtain() -> Path:
        return await download_file_from_url(str(body.url), settings)

    return await _transcribe_source(
        obtain, str(body.url), body, media, transcriber, gate, worker
    )


@router.post(
    "/audio/transcriptions",
    response_model=TranscribeResponse,
    responses=_RESPONSES,
    tags=["openai-compatible"],
)
async def openai_transcriptions(
    file: UploadFile,
    settings: SettingsDep,
    transcriber: TranscriberDep,
    media: MediaDep,
    gate: GateDep,
    worker: WorkerDep,
    model: Annotated[str | None, Form()] = None,  # accepted, server model wins
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form(max_length=2000)] = None,
    response_format: Annotated[OutputFormat, Form()] = OutputFormat.json,
    temperature: Annotated[float | None, Form(ge=0.0, le=1.0)] = None,
    timestamp_granularities: Annotated[list[str] | None, Form()] = None,
) -> Response:
    """OpenAI ``/v1/audio/transcriptions`` shape, so the OpenAI SDK works
    against this server with ``base_url=http://host:8000/api/v1``."""
    granularities = set(timestamp_granularities or [])
    params = TranscribeParams(
        language=language or "auto",
        include_segments=response_format == OutputFormat.verbose_json,
        word_timestamps="word" in granularities,
        initial_prompt=prompt,
        temperature=temperature,
        output_format=response_format,
    )

    async def obtain() -> Path:
        await validate_upload(file, settings)
        return await save_temp_file(file, settings)

    return await _transcribe_source(
        obtain, file.filename or "upload", params, media, transcriber, gate, worker
    )


# ------------------------------------------------------------------ pipeline


async def _transcribe_source(
    obtain: Callable[[], Awaitable[Path]],
    label: str,
    params: TranscribeParams,
    media: MediaService,
    transcriber: TranscriberService,
    gate: JobGate,
    worker: MlxWorker,
) -> Response:
    async with gate.reserve("transcribe") as job:
        input_path = await obtain()
        wav_path = input_path.with_name(f"{input_path.stem}_extracted.wav")
        try:
            result = await _extract_and_transcribe(
                gate, worker, job, media, transcriber, input_path, wav_path, params
            )
        finally:
            cleanup_temp(input_path, wav_path)

    logger.info(
        f"Transcription complete: {label} | "
        f"language={result.language_detected} | "
        f"duration={result.duration_seconds}s | "
        f"processing={result.processing_time_seconds}s"
    )
    return render(result, params)


async def _extract_and_transcribe(
    gate: JobGate,
    worker: MlxWorker,
    job: Job,
    media: MediaService,
    transcriber: TranscriberService,
    input_path: Path,
    wav_path: Path,
    params: TranscribeParams,
) -> TranscribeResponse:
    """Run ffmpeg off the event loop, then whisper on the MLX worker under the
    GPU gate. ffmpeg is CPU-only and runs before the GPU slot is taken."""
    audio = await asyncio.to_thread(media.extract_and_load, input_path, wav_path)
    options = TranscribeOptions(
        language=params.language,
        word_timestamps=params.word_timestamps,
        initial_prompt=params.initial_prompt,
        temperature=(
            params.temperature
            if params.temperature is not None
            else TranscribeOptions().temperature
        ),
        condition_on_previous_text=params.condition_on_previous_text,
    )
    estimate = gate.estimate_transcribe(audio.duration_seconds)
    async with gate.run(job, estimated_seconds=estimate):
        return await worker.run(
            transcriber.transcribe,
            audio.samples,
            duration_seconds=audio.duration_seconds,
            options=options,
        )


def render(result: TranscribeResponse, params: TranscribeParams) -> Response:
    """Serialise the transcript in the requested output format."""
    fmt = params.output_format
    if fmt == OutputFormat.text:
        return PlainTextResponse(result.text)
    if fmt == OutputFormat.srt:
        return PlainTextResponse(
            formats.to_srt(result.segments), media_type="application/x-subrip"
        )
    if fmt == OutputFormat.vtt:
        return PlainTextResponse(formats.to_vtt(result.segments), media_type="text/vtt")

    if fmt == OutputFormat.json and not params.include_segments:
        result = result.model_copy(update={"segments": []})
    return Response(
        content=result.model_dump_json(exclude_none=False),
        media_type="application/json",
    )
