import time

from loguru import logger

from app.config import Settings
from app.core.exceptions import EmbeddingError, ModelNotReadyError


class EmbedderService:
    """Singleton wrapper around mlx-embeddings (Qwen3-Embedding)."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model_repo = settings.embedding_model_repo
        self._model = None
        self._tokenizer = None
        self._ready = False

    def load(self) -> None:
        """Load the embedding model + tokenizer. Called at startup."""
        from mlx_embeddings import load

        logger.info(f"Loading embedding model: {self._model_repo}")
        self._model, self._tokenizer = load(self._model_repo)
        self._ready = True
        logger.info("Embedding model loaded successfully")

    def is_ready(self) -> bool:
        return self._ready

    def _count_tokens(self, texts: list[str]) -> int:
        """Best-effort prompt token count for the usage field."""
        try:
            return sum(len(self._tokenizer.encode(t)) for t in texts)
        except Exception:
            return 0

    def embed(
        self,
        texts: list[str],
        prompt: str | None = None,
    ) -> tuple[list[list[float]], int]:
        """Embed a batch of texts. Returns (vectors, prompt_token_count)."""
        if not self._ready:
            raise ModelNotReadyError()

        from mlx_embeddings import generate

        inputs = [f"{prompt}{t}" for t in texts] if prompt else texts
        start = time.perf_counter()

        try:
            output = generate(self._model, self._tokenizer, texts=inputs)
            vectors = output.text_embeds.tolist()
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

        tokens = self._count_tokens(inputs)
        logger.info(
            f"Embedded {len(inputs)} text(s) | tokens={tokens} | "
            f"processing={round(time.perf_counter() - start, 3)}s"
        )
        return vectors, tokens
