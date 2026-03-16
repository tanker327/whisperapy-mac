class WhisperapyError(Exception):
    """Base exception for whisperapy-mac."""

    def __init__(self, message: str = "An error occurred"):
        self.message = message
        super().__init__(self.message)


class FileTooLargeError(WhisperapyError):
    """File exceeds MAX_FILE_SIZE_MB."""

    def __init__(self, message: str = "File exceeds maximum allowed size"):
        super().__init__(message)


class UnsupportedFormatError(WhisperapyError):
    """Extension or magic bytes not allowed."""

    def __init__(self, message: str = "File type is not supported"):
        super().__init__(message)


class FileValidationError(WhisperapyError):
    """Malformed upload."""

    def __init__(self, message: str = "File validation failed"):
        super().__init__(message)


class AudioExtractionError(WhisperapyError):
    """ffmpeg failed."""

    def __init__(self, message: str = "Audio extraction failed"):
        super().__init__(message)


class TranscriptionError(WhisperapyError):
    """mlx-whisper failed."""

    def __init__(self, message: str = "Transcription failed"):
        super().__init__(message)


class DownloadError(WhisperapyError):
    """File download from URL failed."""

    def __init__(self, message: str = "File download failed"):
        super().__init__(message)


class ModelNotReadyError(WhisperapyError):
    """Model not yet loaded at startup."""

    def __init__(self, message: str = "Model is not ready"):
        super().__init__(message)
