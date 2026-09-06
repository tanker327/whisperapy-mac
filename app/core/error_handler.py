from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.exceptions import ServiceBusyError, WhisperapyError


def error_response(
    request: Request, exc: WhisperapyError, status_code: int | None = None
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    content: dict = {
        "error": type(exc).__name__,
        "message": exc.message,
        "request_id": request_id,
    }
    headers: dict[str, str] = {}
    if isinstance(exc, ServiceBusyError):
        content["retry_after_seconds"] = exc.retry_after
        headers["Retry-After"] = str(exc.retry_after)
    return JSONResponse(
        status_code=status_code or exc.status_code, content=content, headers=headers
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(WhisperapyError)
    async def whisperapy_error_handler(
        request: Request, exc: WhisperapyError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"{type(exc).__name__}: {exc.message} | request_id={request_id}")
        return error_response(request, exc)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(f"Unhandled error: {exc} | request_id={request_id}")
        # This handler runs in Starlette's outermost ServerErrorMiddleware,
        # outside RequestContextMiddleware, so the header is set here by hand.
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )
