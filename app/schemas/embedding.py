from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    input: str | list[str]
    model: str | None = None  # echoed back; server model is authoritative
    encoding_format: Literal["float", "base64"] = "float"
    # Matryoshka truncation of the returned vector (Qwen3-Embedding supports it).
    dimensions: int | None = Field(default=None, ge=32, le=4096)
    # Extension: optional Qwen3 instruction/prompt to prepend (e.g. for queries).
    prompt: str | None = Field(default=None, max_length=2000)

    @field_validator("input")
    @classmethod
    def _reject_empty_input(cls, v: str | list[str]) -> str | list[str]:
        """Reject empty/blank input as a 422 rather than a 500 downstream."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("input must not be empty")
            return v
        if len(v) == 0:
            raise ValueError("input must not be an empty list")
        if any(not t.strip() for t in v):
            raise ValueError("input items must not be empty")
        return v

    @property
    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float] | str  # str when encoding_format == "base64"


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData] = Field(default_factory=list)
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    # Extension: how many inputs were cut to EMBEDDING_MAX_TOKENS (0 unless
    # EMBEDDING_TRUNCATE is enabled).
    truncated: int = 0


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "whisperapy-mac"
    task: str


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
