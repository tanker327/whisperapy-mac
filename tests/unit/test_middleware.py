from app.core.middleware import RequestContextMiddleware


async def test_non_http_scopes_pass_straight_through():
    seen = {}

    async def inner(scope, receive, send):
        seen["scope"] = scope

    mw = RequestContextMiddleware(inner)
    scope = {"type": "lifespan"}
    await mw(scope, None, None)
    assert seen["scope"] is scope
    assert "state" not in scope  # untouched: no request id for lifespan


async def test_http_scope_gets_request_id_and_headers():
    sent = []

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "headers": [(b"x-request-id", b"abc")]}
    await RequestContextMiddleware(inner)(scope, None, send)
    assert scope["state"]["request_id"] == "abc"
    headers = dict(sent[0]["headers"])
    assert headers[b"x-request-id"] == b"abc"
    assert float(headers[b"x-processing-time"]) >= 0
