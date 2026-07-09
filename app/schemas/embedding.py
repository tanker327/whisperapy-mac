from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    input: str | list[str]
    model: str | None = None  # echoed back; server model is authoritative
    encoding_format: str = "float"
    # Extension: optional Qwen3 instruction/prompt to prepend (e.g. for queries).
    prompt: str | None = None

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


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData] = Field(default_factory=list)
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
