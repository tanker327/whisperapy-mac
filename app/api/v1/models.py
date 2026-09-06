from fastapi import APIRouter

from app.dependencies import SettingsDep
from app.schemas.embedding import ModelInfo, ModelList

router = APIRouter(prefix="/models", tags=["openai-compatible"])


@router.get("", response_model=ModelList)
async def list_models(settings: SettingsDep) -> ModelList:
    """Minimal OpenAI ``/v1/models`` so SDKs that probe it on startup work."""
    return ModelList(
        data=[
            ModelInfo(id=settings.model_repo, task="transcription"),
            ModelInfo(id=settings.embedding_model_repo, task="embedding"),
        ]
    )
