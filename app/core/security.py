"""Optional API-key authentication for the v1 router.

Enabled by setting ``API_KEY``. Clients send either
``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. Health endpoints are
mounted outside the v1 router and stay open so monitors keep working.
"""

import hmac

from fastapi import Request

from app.core.exceptions import AuthenticationError
from app.dependencies import SettingsDep


def _presented_key(request: Request) -> str | None:
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


async def require_api_key(request: Request, settings: SettingsDep) -> None:
    if settings.api_key is None:
        return
    presented = _presented_key(request)
    if presented is None or not hmac.compare_digest(presented, settings.api_key):
        raise AuthenticationError()
