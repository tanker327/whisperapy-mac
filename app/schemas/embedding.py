from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    input: str | list[str]
    model: str | None = None  # echoed back; server model is authoritative
    encoding_format: str = "float"
    # Extension: optional Qwen3 instruction/prompt to prepend (e.g. for queries).
    prompt: str | None = None


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
