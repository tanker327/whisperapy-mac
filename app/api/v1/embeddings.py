from fastapi import APIRouter, Depends

from app.config import Settings
from app.core.gate import JobGate
from app.core.mlx_worker import MlxWorker
from app.dependencies import get_embedder, get_gate, get_mlx_worker, get_settings
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
    gate: JobGate = Depends(get_gate),
    worker: MlxWorker = Depends(get_mlx_worker),
) -> EmbeddingResponse:
    """OpenAI-compatible text embeddings via Qwen3-Embedding."""
    texts = [body.input] if isinstance(body.input, str) else body.input
    async with gate.reserve("embed") as job, gate.run(job):
        vectors, tokens = await worker.run(embedder.embed, texts, prompt=body.prompt)
    data = [EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(vectors)]
    return EmbeddingResponse(
        data=data,
        model=settings.embedding_model_repo,
        usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
    )
