from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.exceptions import WhisperapyError


def register_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(WhisperapyError)
    async def whisperapy_error_handler(
        request: Request, exc: WhisperapyError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"{type(exc).__name__}: {exc.message} | request_id={request_id}")
        status_code = _get_status_code(exc)
        return JSONResponse(
            status_code=status_code,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(f"Unhandled error: {exc} | request_id={request_id}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "request_id": request_id,
            },
        )


def _get_status_code(exc: WhisperapyError) -> int:
    from app.core.exceptions import (
        AudioExtractionError,
        DownloadError,
        FileTooLargeError,
        FileValidationError,
        ModelNotReadyError,
        TranscriptionError,
        UnsupportedFormatError,
    )

    status_map = {
        FileTooLargeError: 413,
        UnsupportedFormatError: 415,
        FileValidationError: 422,
        DownloadError: 422,
        AudioExtractionError: 500,
        TranscriptionError: 500,
        ModelNotReadyError: 503,
    }
    return status_map.get(type(exc), 500)
