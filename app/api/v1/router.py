from fastapi import APIRouter

from app.api.v1.transcribe import router as transcribe_router

router = APIRouter(prefix="/api/v1")
router.include_router(transcribe_router)
