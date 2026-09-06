import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.config import Settings
from app.core.exceptions import EmbeddingError, InputTooLongError, ModelNotReadyError


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    prompt_tokens: int
    truncated: int  # number of inputs that were cut to embedding_max_tokens


class EmbedderService:
    """Singleton wrapper around mlx-embeddings (Qwen3-Embedding)."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model_repo = settings.embedding_model_repo
        self._model: Any = None
        self._tokenizer: Any = None
        self._ready = False

    def load(self) -> None:
        """Load the embedding model + tokenizer. Called at startup."""
        from mlx_embeddings import load

        logger.info(f"Loading embedding model: {self._model_repo}")
        start = time.perf_counter()
        self._model, self._tokenizer = load(self._model_repo)
        self._ready = True
        logger.info(f"Embedding model loaded in {time.perf_counter() - start:.1f}s")

    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------ tokenizing

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Per-input token counts. Zero when the tokenizer cannot be used."""
        if self._tokenizer is None:
            return [0] * len(texts)
        try:
            return [len(self._tokenizer.encode(t)) for t in texts]
        except Exception:
            return [0] * len(texts)

    def check_lengths(self, counts: list[int]) -> int:
        """Enforce the token limit; return how many inputs will be truncated.

        Raises ``InputTooLongError`` unless truncation is enabled in settings.
        """
        limit = self._settings.embedding_max_tokens
        over = [i for i, n in enumerate(counts) if n > limit]
        if over and not self._settings.embedding_truncate:
            raise InputTooLongError(
                f"Input {over[0]} is {counts[over[0]]} tokens; the limit is {limit}. "
                "Split the text or enable EMBEDDING_TRUNCATE."
            )
        return len(over)

    # -------------------------------------------------------------- embedding

    def embed(
        self,
        texts: list[str],
        prompt: str | None = None,
        dimensions: int | None = None,
    ) -> EmbedResult:
        """Embed a batch of texts on the MLX thread.

        Inputs are processed in chunks of ``embedding_batch_size`` so a large
        request cannot allocate one enormous padded batch on the GPU.
        """
        if not self._ready:
            raise ModelNotReadyError()

        from mlx_embeddings import generate

        inputs = [f"{prompt}{t}" for t in texts] if prompt else list(texts)
        counts = self.count_tokens(inputs)
        truncated = self.check_lengths(counts)
        max_len = self._settings.embedding_max_tokens
        batch = max(1, self._settings.embedding_batch_size)
        start = time.perf_counter()

        vectors: list[list[float]] = []
        try:
            for i in range(0, len(inputs), batch):
                output: Any = generate(
                    self._model,
                    self._tokenizer,
                    texts=inputs[i : i + batch],
                    max_length=max_len,
                    truncation=True,
                )
                vectors.extend(output.text_embeds.tolist())
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

        if dimensions is not None:
            vectors = [_shrink(v, dimensions) for v in vectors]

        # Usage reflects what was embedded, not what was sent.
        prompt_tokens = sum(min(n, max_len) for n in counts)
        logger.info(
            f"Embedded {len(inputs)} text(s) | tokens={prompt_tokens} | "
            f"truncated={truncated} | "
            f"processing={round(time.perf_counter() - start, 3)}s"
        )
        return EmbedResult(
            vectors=vectors, prompt_tokens=prompt_tokens, truncated=truncated
        )


def _shrink(vector: list[float], dimensions: int) -> list[float]:
    """Matryoshka truncation: keep the leading dims and re-normalise."""
    head = vector[:dimensions]
    norm = sum(x * x for x in head) ** 0.5
    return [x / norm for x in head] if norm else head
