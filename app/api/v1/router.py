from fastapi import APIRouter, Depends

from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.models import router as models_router
from app.api.v1.transcribe import router as transcribe_router
from app.core.security import require_api_key

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
router.include_router(transcribe_router)
router.include_router(embeddings_router)
router.include_router(models_router)
