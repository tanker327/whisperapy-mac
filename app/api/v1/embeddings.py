from fastapi import APIRouter, Depends

from app.config import Settings
from app.dependencies import get_embedder, get_settings
from app.schemas.embedding import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    Usage,
)
from app.services.embedder import EmbedderService

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("", response_model=EmbeddingResponse)
async def create_embeddings(
    body: EmbeddingRequest,
    settings: Settings = Depends(get_settings),
    embedder: EmbedderService = Depends(get_embedder),
) -> EmbeddingResponse:
    """OpenAI-compatible text embeddings via Qwen3-Embedding."""
    texts = [body.input] if isinstance(body.input, str) else body.input
    vectors, tokens = embedder.embed(texts, prompt=body.prompt)
    data = [EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(vectors)]
    return EmbeddingResponse(
        data=data,
        model=settings.embedding_model_repo,
        usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
    )
