"""Request-ID and timing as one pure-ASGI middleware.

Starlette discourages ``BaseHTTPMiddleware`` (it breaks contextvars across
``call_next`` and buffers streaming responses). A plain ASGI callable avoids
both problems and costs a few more lines.
"""

import time
import uuid

from loguru import logger
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = b"x-request-id"
PROCESSING_TIME_HEADER = b"x-processing-time"


class RequestContextMiddleware:
    """Assign (or honour) an ``X-Request-ID``, time the request, and bind the
    id into loguru's context for every log line emitted while handling it."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers", [])).get(REQUEST_ID_HEADER)
        request_id = incoming.decode("latin-1")[:64] if incoming else str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        start = time.perf_counter()

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER, request_id.encode("latin-1")))
                duration = time.perf_counter() - start
                headers.append((PROCESSING_TIME_HEADER, f"{duration:.4f}".encode()))
                message["headers"] = headers
            await send(message)

        with logger.contextualize(request_id=request_id):
            await self.app(scope, receive, send_with_headers)
