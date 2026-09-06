"""Domain errors. Each carries the HTTP status it maps to, so the global
handler in ``app/core/error_handler.py`` needs no lookup table."""

from typing import ClassVar


class WhisperapyError(Exception):
    """Base exception for whisperapy-mac."""

    status_code: ClassVar[int] = 500
    default_message: ClassVar[str] = "An error occurred"

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


class FileTooLargeError(WhisperapyError):
    """File exceeds MAX_FILE_SIZE_MB."""

    status_code = 413
    default_message = "File exceeds maximum allowed size"


class UnsupportedFormatError(WhisperapyError):
    """Extension or magic bytes not allowed."""

    status_code = 415
    default_message = "File type is not supported"


class FileValidationError(WhisperapyError):
    """Malformed upload."""

    status_code = 422
    default_message = "File validation failed"


class DownloadError(WhisperapyError):
    """File download from URL failed."""

    status_code = 422
    default_message = "File download failed"


class ForbiddenUrlError(WhisperapyError):
    """URL points at a loopback / private / link-local address."""

    status_code = 422
    default_message = "URL host is not allowed"


class InvalidRequestError(WhisperapyError):
    """Request is well-formed but violates a configured limit."""

    status_code = 422
    default_message = "Invalid request"


class InputTooLongError(WhisperapyError):
    """Embedding input exceeds EMBEDDING_MAX_TOKENS."""

    status_code = 422
    default_message = "Input exceeds the maximum token length"


class AuthenticationError(WhisperapyError):
    """Missing or wrong API key."""

    status_code = 401
    default_message = "Missing or invalid API key"


class AudioExtractionError(WhisperapyError):
    """ffmpeg failed."""

    status_code = 500
    default_message = "Audio extraction failed"


class TranscriptionError(WhisperapyError):
    """mlx-whisper failed."""

    status_code = 500
    default_message = "Transcription failed"


class EmbeddingError(WhisperapyError):
    """Embedding generation failed."""

    status_code = 500
    default_message = "Embedding failed"


class ModelNotReadyError(WhisperapyError):
    """Model not yet loaded at startup."""

    status_code = 503
    default_message = "Model is not ready"


class ServiceBusyError(WhisperapyError):
    """All GPU job slots are taken; the client should retry later."""

    status_code = 503
    default_message = "Server is busy processing another request. Please retry later."

    def __init__(self, message: str | None = None, retry_after: int = 30):
        self.retry_after = retry_after
        super().__init__(message)
