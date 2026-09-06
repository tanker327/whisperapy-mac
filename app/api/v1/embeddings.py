import asyncio
import base64
import struct

from fastapi import APIRouter

from app.core.exceptions import InvalidRequestError
from app.dependencies import EmbedderDep, GateDep, SettingsDep, WorkerDep
from app.schemas.embedding import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    Usage,
)

router = APIRouter(prefix="/embeddings", tags=["embeddings", "openai-compatible"])


def _encode(vector: list[float], fmt: str) -> list[float] | str:
    if fmt == "base64":
        # OpenAI encodes float32 little-endian.
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
    return vector


@router.post("", response_model=EmbeddingResponse)
async def create_embeddings(
    body: EmbeddingRequest,
    settings: SettingsDep,
    embedder: EmbedderDep,
    gate: GateDep,
    worker: WorkerDep,
) -> EmbeddingResponse:
    """OpenAI-compatible text embeddings via Qwen3-Embedding."""
    texts = body.texts
    if len(texts) > settings.embedding_max_batch:
        raise InvalidRequestError(
            f"input has {len(texts)} items; the limit is {settings.embedding_max_batch}"
        )

    # Token counting is CPU-only (HF tokenizer, not MLX), so it runs in an
    # ordinary thread rather than queueing behind whatever holds the GPU. It
    # happens before the slot is taken so the gate can estimate this job.
    counts = await asyncio.to_thread(embedder.count_tokens, texts)
    estimate = gate.estimate_embed(sum(counts))

    async with gate.reserve("embed") as job, gate.run(job, estimated_seconds=estimate):
        result = await worker.run(
            embedder.embed, texts, prompt=body.prompt, dimensions=body.dimensions
        )

    data = [
        EmbeddingData(index=i, embedding=_encode(vec, body.encoding_format))
        for i, vec in enumerate(result.vectors)
    ]
    return EmbeddingResponse(
        data=data,
        model=settings.embedding_model_repo,
        usage=Usage(
            prompt_tokens=result.prompt_tokens, total_tokens=result.prompt_tokens
        ),
        truncated=result.truncated,
    )
