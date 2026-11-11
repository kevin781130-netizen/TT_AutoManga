"""Contract tests for the InvokeAI HTTP client; no live InvokeAI process required."""
import base64
import sys
from pathlib import Path

import httpx
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT.parent))

from backend.core.invokeai_client import (
    InvokeAIClient,
    InvokeAIHTTPError,
    InvokeAIJobError,
    InvokeAIProtocolError,
)

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def client_for(handler, **kwargs):
    return InvokeAIClient(
        transport=httpx.MockTransport(handler),
        poll_interval=0,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_connection_uses_public_version_endpoint():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"version": "5.12.0"})

    result = await client_for(handler).check_connection()
    assert result == {"connected": True, "status": "ok", "version": "5.12.0"}
    assert seen == [("GET", "/api/v1/app/version")]


@pytest.mark.asyncio
async def test_model_and_lora_discovery_filters_by_metadata():
    def handler(request):
        assert request.url.path == "/api/v2/models/"
        return httpx.Response(200, json={"models": [
            {"key": "qwen-main", "name": "Qwen Image 2512", "base": "qwen-image", "type": "main"},
            {"key": "qwen-lora", "name": "Character", "base": "qwen-image", "type": "lora"},
            {"key": "sdxl-main", "name": "SDXL", "base": "sdxl", "type": "main"},
            {"key": "flux-lora", "name": "Flux Style", "base": "flux", "type": "lora"},
        ]})

    client = client_for(handler)
    assert [m["key"] for m in await client.list_models()] == ["qwen-main"]
    assert [m["key"] for m in await client.list_loras()] == ["qwen-lora"]


@pytest.mark.asyncio
async def test_model_response_shape_is_validated():
    def handler(request):
        return httpx.Response(200, json={"unexpected": []})

    with pytest.raises(InvokeAIProtocolError, match="models array"):
        await client_for(handler).get_model_records()


@pytest.mark.asyncio
async def test_enqueue_graph_uses_batch_envelope_and_returns_item_ids():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = request.read()
        return httpx.Response(201, json={
            "queue_id": "default",
            "enqueued": 1,
            "requested": 1,
            "item_ids": [42],
        })

    client = client_for(handler)
    result = await client.enqueue_graph(
        {
            "nodes": {"n1": {"id": "n1", "type": "test"}},
            "edges": [],
        },
        destination="tt-automanga:panel:panel1",
    )
    assert result["item_ids"] == [42]
    assert seen["path"] == "/api/v1/queue/default/enqueue_batch"
    body = __import__("json").loads(seen["body"])
    assert body["batch"]["runs"] == 1
    assert body["batch"]["origin"] == "tt-automanga"
    assert body["batch"]["destination"] == "tt-automanga:panel:panel1"
    assert body["batch"]["graph"]["nodes"]["n1"]["type"] == "test"
    assert body["prepend"] is False


@pytest.mark.asyncio
async def test_queue_polling_returns_completed_item():
    responses = iter([
        {"item_id": 7, "status": "pending"},
        {"item_id": 7, "status": "in_progress"},
        {"item_id": 7, "status": "completed", "session": {"results": {}}},
    ])

    def handler(request):
        return httpx.Response(200, json=next(responses))

    result = await client_for(handler).wait_for_queue_item(7, timeout=1)
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_queue_failure_is_not_reported_as_success():
    def handler(request):
        return httpx.Response(200, json={
            "item_id": 8, "status": "failed", "error_message": "GPU out of memory"
        })

    with pytest.raises(InvokeAIJobError, match="GPU out of memory"):
        await client_for(handler).wait_for_queue_item(8, timeout=1)


@pytest.mark.asyncio
async def test_queue_item_must_match_requested_id():
    def handler(request):
        return httpx.Response(200, json={"item_id": 99, "status": "completed"})

    with pytest.raises(InvokeAIProtocolError, match="requested item"):
        await client_for(handler).get_queue_item(7)


@pytest.mark.asyncio
async def test_download_image_requires_valid_image_bytes():
    def handler(request):
        assert request.url.path == "/api/v1/images/i/generated.png/full"
        return httpx.Response(200, content=PNG_1PX, headers={"content-type": "image/png"})

    assert await client_for(handler).download_image("generated.png") == PNG_1PX


@pytest.mark.asyncio
async def test_download_rejects_corrupt_image_bytes():
    def handler(request):
        return httpx.Response(200, content=b"not an image")

    with pytest.raises(InvokeAIProtocolError, match="invalid or corrupt"):
        await client_for(handler).download_image("generated.png")


@pytest.mark.asyncio
async def test_download_rejects_path_traversal():
    client = client_for(lambda request: httpx.Response(200, content=PNG_1PX))
    with pytest.raises(ValueError, match="plain InvokeAI image filename"):
        await client.download_image("../secret.png")


def test_base_url_rejects_credentials_and_non_http_schemes():
    with pytest.raises(ValueError):
        InvokeAIClient("file:///tmp/invoke")
    with pytest.raises(ValueError):
        InvokeAIClient("http://user:password@127.0.0.1:9090")
