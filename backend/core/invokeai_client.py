"""Small, version-aware HTTP client for a locally hosted InvokeAI server.

This module deliberately owns transport and queue lifecycle only. Graph construction
is kept in invokeai_qwen2512_graph.py so the InvokeAI API contract and the model
architecture contract can be tested independently.
"""
from __future__ import annotations

import asyncio
import io
import re
import time
import uuid
from typing import Any, Mapping
from urllib.parse import quote, urlsplit

import httpx
from PIL import Image


class InvokeAIError(RuntimeError):
    """Base exception for InvokeAI integration failures."""


class InvokeAIConnectionError(InvokeAIError):
    """The configured InvokeAI endpoint could not be reached."""


class InvokeAIHTTPError(InvokeAIError):
    """InvokeAI returned an unsuccessful HTTP response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        if status_code in (401, 403):
            message = (
                f"InvokeAI rejected the request (HTTP {status_code}); "
                "check the InvokeAI authentication configuration."
            )
        else:
            message = f"InvokeAI returned HTTP {status_code}: {detail}"
        super().__init__(message)


class InvokeAIProtocolError(InvokeAIError):
    """InvokeAI returned a payload that does not match the expected API contract."""


class InvokeAIJobError(InvokeAIError):
    """A queued InvokeAI generation failed or was canceled."""

    def __init__(self, item_id: int, status: str, detail: str = ""):
        self.item_id = item_id
        self.status = status
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"InvokeAI queue item {item_id} ended with status {status}{suffix}")


class InvokeAITimeoutError(InvokeAIError):
    """An InvokeAI queue item did not finish within the configured timeout."""


class InvokeAIClient:
    """Async HTTP client for InvokeAI v1 queue, v1 images and v2 model APIs."""

    _QWEN_BASE = "qwen-image"
    _SUPPORTED_QUEUE_STATUSES = {"pending", "in_progress", "waiting", "completed", "failed", "canceled"}

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9090",
        *,
        request_timeout: float = 30.0,
        generation_timeout: float = 900.0,
        poll_interval: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = self._validate_base_url(base_url)
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")
        if generation_timeout <= 0:
            raise ValueError("generation_timeout must be greater than zero")
        if poll_interval < 0:
            raise ValueError("poll_interval cannot be negative")
        self.request_timeout = float(request_timeout)
        self.generation_timeout = float(generation_timeout)
        self.poll_interval = float(poll_interval)
        self._transport = transport

    @staticmethod
    def _validate_base_url(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("InvokeAI base_url is required")
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("InvokeAI base_url must be an absolute http:// or https:// URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("InvokeAI base_url cannot contain credentials, query, or fragment")
        return value.strip().rstrip("/")

    def _new_client(self, timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.request_timeout if timeout is None else timeout,
            follow_redirects=False,
            transport=self._transport,
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            async with self._new_client(timeout) as client:
                response = await client.request(method, path, params=params, json=json_body)
        except httpx.RequestError as exc:
            raise InvokeAIConnectionError(
                f"Could not reach InvokeAI at {self.base_url}: {exc.__class__.__name__}"
            ) from exc

        if response.status_code >= 400:
            detail = response.text[:500].replace("\\n", " ")
            raise InvokeAIHTTPError(response.status_code, detail)

        try:
            payload = response.json()
        except ValueError as exc:
            raise InvokeAIProtocolError(
                f"InvokeAI {method} {path} returned non-JSON content"
            ) from exc
        if not isinstance(payload, dict):
            raise InvokeAIProtocolError(
                f"InvokeAI {method} {path} returned {type(payload).__name__}; expected an object"
            )
        return payload

    async def check_connection(self) -> dict[str, Any]:
        """Check the public version endpoint; do not infer health from TCP alone."""
        try:
            payload = await self._request_json("GET", "/api/v1/app/version")
            version = payload.get("version")
            if not isinstance(version, str) or not version.strip():
                raise InvokeAIProtocolError("InvokeAI version response did not contain a version")
            return {"connected": True, "status": "ok", "version": version}
        except InvokeAIError as exc:
            return {"connected": False, "status": str(exc)}

    async def get_model_records(self) -> list[dict[str, Any]]:
        """Return all model records from InvokeAI's v2 model manager."""
        payload = await self._request_json("GET", "/api/v2/models/")
        records = payload.get("models")
        if not isinstance(records, list):
            raise InvokeAIProtocolError("InvokeAI model response is missing the models array")
        if any(not isinstance(record, dict) for record in records):
            raise InvokeAIProtocolError("InvokeAI model response contains a non-object model record")
        return records

    async def list_models(self, base_model: str = _QWEN_BASE) -> list[dict[str, Any]]:
        """List installed main models for one architecture, defaulting to Qwen Image."""
        records = await self.get_model_records()
        return [
            record for record in records
            if record.get("base") == base_model and record.get("type") == "main"
        ]

    async def list_loras(self, base_model: str = _QWEN_BASE) -> list[dict[str, Any]]:
        """List installed LoRAs for one architecture, defaulting to Qwen Image."""
        records = await self.get_model_records()
        return [
            record for record in records
            if record.get("base") == base_model and record.get("type") == "lora"
        ]

    async def get_model(self, model_key: str) -> dict[str, Any]:
        if not isinstance(model_key, str) or not model_key.strip():
            raise ValueError("model_key is required")
        for record in await self.get_model_records():
            if record.get("key") == model_key:
                return record
        raise InvokeAIError(f"InvokeAI model key not found: {model_key}")

    async def enqueue_graph(
        self,
        graph: Mapping[str, Any],
        *,
        origin: str = "tt-automanga",
        project_id: str | None = None,
        batch_id: str | None = None,
        idempotency_key: str | None = None,
        destination: str | None = None,
        queue_id: str = "default",
    ) -> dict[str, Any]:
        """Submit one executable graph to InvokeAI's session queue."""
        if not isinstance(graph, Mapping):
            raise ValueError("graph must be a mapping")
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, (dict, list)) or not nodes:
            raise ValueError("graph.nodes must be a non-empty object or array")
        if not isinstance(edges, list):
            raise ValueError("graph.edges must be an array")

        batch: dict[str, Any] = {
            "batch_id": batch_id or uuid.uuid4().hex,
            "graph": dict(graph),
            "runs": 1,
            "origin": origin,
        }
        if project_id:
            batch["project_id"] = project_id
        if idempotency_key:
            batch["idempotency_key"] = idempotency_key
        if destination:
            batch["destination"] = destination

        payload = await self._request_json(
            "POST",
            f"/api/v1/queue/{quote(queue_id, safe='')}/enqueue_batch",
            json_body={"batch": batch, "prepend": False},
            timeout=max(self.request_timeout, 60.0),
        )
        item_ids = payload.get("item_ids")
        if not isinstance(item_ids, list) or not item_ids or any(
            isinstance(item_id, bool) or not isinstance(item_id, int) or item_id <= 0
            for item_id in item_ids
        ):
            raise InvokeAIProtocolError("InvokeAI enqueue response did not contain valid item_ids")
        return payload

    async def get_queue_item(self, item_id: int, queue_id: str = "default") -> dict[str, Any]:
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id <= 0:
            raise ValueError("item_id must be a positive integer")
        payload = await self._request_json(
            "GET",
            f"/api/v1/queue/{quote(queue_id, safe='')}/i/{item_id}",
        )
        if payload.get("item_id") != item_id:
            raise InvokeAIProtocolError(
                f"InvokeAI returned queue item {payload.get('item_id')!r} for requested item {item_id}"
            )
        status = payload.get("status")
        if status not in self._SUPPORTED_QUEUE_STATUSES:
            raise InvokeAIProtocolError(f"InvokeAI returned an unknown queue status: {status!r}")
        return payload

    async def wait_for_queue_item(
        self,
        item_id: int,
        *,
        queue_id: str = "default",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Poll a queue item until it completes or reaches a terminal failure state."""
        effective_timeout = self.generation_timeout if timeout is None else float(timeout)
        if effective_timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        deadline = time.monotonic() + effective_timeout

        while True:
            item = await self.get_queue_item(item_id, queue_id=queue_id)
            status = item["status"]
            if status == "completed":
                return item
            if status in {"failed", "canceled"}:
                detail = item.get("error_message") or item.get("error_type") or ""
                raise InvokeAIJobError(item_id, status, str(detail))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InvokeAITimeoutError(
                    f"InvokeAI queue item {item_id} did not complete within {effective_timeout:.1f}s"
                )
            await asyncio.sleep(min(self.poll_interval, remaining))

    async def download_image(self, image_name: str) -> bytes:
        """Download and verify a generated image using InvokeAI's media endpoint."""
        if not isinstance(image_name, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", image_name):
            raise ValueError("image_name must be a plain InvokeAI image filename")
        if image_name in {".", ".."}:
            raise ValueError("invalid image_name")

        try:
            async with self._new_client(timeout=max(self.request_timeout, 60.0)) as client:
                response = await client.get(
                    f"/api/v1/images/i/{quote(image_name, safe='')}/full"
                )
        except httpx.RequestError as exc:
            raise InvokeAIConnectionError(
                f"Could not download image from InvokeAI: {exc.__class__.__name__}"
            ) from exc

        if response.status_code >= 400:
            raise InvokeAIHTTPError(response.status_code, response.text[:500])
        if not response.content:
            raise InvokeAIProtocolError("InvokeAI returned an empty image")

        try:
            with Image.open(io.BytesIO(response.content)) as image:
                image.verify()
        except Exception as exc:
            raise InvokeAIProtocolError("InvokeAI returned an invalid or corrupt image") from exc
        return response.content
