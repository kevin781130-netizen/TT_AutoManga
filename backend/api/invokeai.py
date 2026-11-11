"""InvokeAI Qwen Image orchestration API.

Submits a native Qwen Image graph, tracks its InvokeAI queue item, and persists
the completed image into the selected TT AutoManga project panel.
"""
from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Mapping
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field
from PIL import Image

from ..core import config
from ..core.invokeai_client import (
    InvokeAIClient,
    InvokeAIError,
    InvokeAIProtocolError,
)
from ..core.invokeai_qwen2512_graph import (
    QWEN_IMAGE_OUTPUT_NODE_ID,
    QwenImageGraphError,
    build_qwen_image_2512_graph,
)
from ..storage.manager import storage, InvalidProjectIdError
from ..storage.asset_store import asset_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/invokeai", tags=["invokeai"])

_JOB_DESTINATION_PREFIX = "tt-automanga:panel:"
_job_locks: dict[int, asyncio.Lock] = {}


class InvokeAILoRASelection(BaseModel):
    model_key: str = Field(min_length=1, max_length=255)
    weight: float = Field(default=1.0, allow_inf_nan=False)


class InvokeAIGenerateRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    panel_id: str = Field(min_length=1, max_length=255)
    prompt: str = Field(min_length=1, max_length=20000)
    model_key: str = Field(min_length=1, max_length=255)
    loras: list[InvokeAILoRASelection] = Field(default_factory=list)
    width: int = Field(default=1024, gt=0, multiple_of=16, le=4096)
    height: int = Field(default=1024, gt=0, multiple_of=16, le=4096)
    steps: int = Field(default=40, gt=0, le=200)
    cfg_scale: float = Field(default=4.0, allow_inf_nan=False, ge=0, le=30)
    seed: int = 0
    component_source_key: str | None = Field(default=None, max_length=255)
    vae_model_key: str | None = Field(default=None, max_length=255)
    qwen_vl_encoder_model_key: str | None = Field(default=None, max_length=255)
    quantization: Literal["none", "int8", "nf4"] = "none"
    tiled_decode: bool = False
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)


def get_invokeai_client() -> InvokeAIClient:
    """Create a client from the current persisted InvokeAI settings."""
    settings = config.settings.invokeai
    return InvokeAIClient(
        base_url=settings.address,
        request_timeout=settings.request_timeout,
        generation_timeout=settings.generation_timeout,
    )


def _public_model_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only the fields needed for model selection and graph construction."""
    allowed = ("key", "hash", "name", "base", "type", "format")
    return {field: record[field] for field in allowed if field in record}


def _selected_record(
    records: list[dict[str, Any]],
    model_key: str,
    *,
    label: str,
) -> dict[str, Any]:
    for record in records:
        if record.get("key") == model_key:
            return record
    raise HTTPException(status_code=400, detail=f"Selected {label} was not found in InvokeAI: {model_key}")


def _validate_project_panel(project_id: str, panel_id: str) -> dict[str, Any]:
    try:
        project = storage.load_project(project_id)
    except InvalidProjectIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not project:
        raise HTTPException(status_code=404, detail="TT AutoManga project not found")

    panels = project.get("panels")
    if not isinstance(panels, list) or not any(
        isinstance(panel, dict) and panel.get("id") == panel_id for panel in panels
    ):
        raise HTTPException(status_code=404, detail="Panel not found in the selected project")
    return project


def _extract_image_name(queue_item: Mapping[str, Any]) -> str:
    session = queue_item.get("session")
    if not isinstance(session, Mapping):
        raise InvokeAIProtocolError("Completed InvokeAI queue item has no session object")

    results = session.get("results")
    if not isinstance(results, Mapping):
        raise InvokeAIProtocolError("Completed InvokeAI session has no node results")

    output = results.get(QWEN_IMAGE_OUTPUT_NODE_ID)
    if not isinstance(output, Mapping):
        raise InvokeAIProtocolError(
            f"Completed InvokeAI session has no output for node {QWEN_IMAGE_OUTPUT_NODE_ID}"
        )

    image = output.get("image")
    if not isinstance(image, Mapping):
        raise InvokeAIProtocolError("Qwen Image decode node did not return an image field")

    image_name = image.get("image_name")
    if not isinstance(image_name, str) or not image_name:
        raise InvokeAIProtocolError("Qwen Image decode node returned no image_name")
    return image_name


def _find_job_asset(item_id: int) -> dict[str, Any] | None:
    for asset_id in asset_store.ids():
        entry = asset_store.get(asset_id)
        if (
            entry
            and entry.get("source") == "invokeai"
            and entry.get("invokeai_item_id") == item_id
        ):
            return entry
    return None


def _image_extension(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = image.format
            image.verify()
    except Exception as exc:
        raise InvokeAIProtocolError("Downloaded InvokeAI output is not a valid image") from exc

    extension = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}.get(image_format or "")
    if extension is None:
        raise InvokeAIProtocolError(f"Unsupported InvokeAI image format: {image_format!r}")
    return extension


async def _persist_completed_job(
    client: InvokeAIClient,
    queue_item: Mapping[str, Any],
    item_id: int,
) -> dict[str, str]:
    """Idempotently save a completed queue result and attach it to its project panel."""
    project_id = queue_item.get("project_id")
    destination = queue_item.get("destination")
    if not isinstance(project_id, str) or not project_id:
        raise InvokeAIProtocolError("InvokeAI queue item is missing its TT AutoManga project_id")
    if not isinstance(destination, str) or not destination.startswith(_JOB_DESTINATION_PREFIX):
        raise InvokeAIProtocolError("InvokeAI queue item is missing its TT AutoManga panel destination")

    panel_id = destination[len(_JOB_DESTINATION_PREFIX):]
    if not panel_id:
        raise InvokeAIProtocolError("InvokeAI queue item contains an empty panel destination")

    lock = _job_locks.setdefault(item_id, asyncio.Lock())
    async with lock:
        project = _validate_project_panel(project_id, panel_id)

        existing = _find_job_asset(item_id)
        if existing is not None and existing.get("project_id") == project_id:
            asset_id = existing.get("asset_id")
            if isinstance(asset_id, str) and asset_store.path_for(asset_id) is not None:
                image_url = asset_store.url_for(asset_id)
            else:
                if isinstance(asset_id, str):
                    asset_store.remove(asset_id)
                existing = None
        else:
            if existing is not None:
                raise InvokeAIProtocolError("Queue item is already associated with a different project asset")
            image_url = ""

        if existing is None:
            image_name = _extract_image_name(queue_item)
            image_bytes = await client.download_image(image_name)
            extension = _image_extension(image_bytes)
            entry = asset_store.save_bytes(
                image_bytes,
                ext=extension,
                project_id=project_id,
                kind="qwen-generated",
                meta={
                    "source": "invokeai",
                    "invokeai_item_id": item_id,
                    "panel_id": panel_id,
                },
            )
            asset_id = entry["asset_id"]
            image_url = asset_store.url_for(asset_id)

        panel = next(
            panel for panel in project["panels"]
            if isinstance(panel, dict) and panel.get("id") == panel_id
        )
        panel["image"] = image_url
        storage.save_project(project)
        return {"asset_id": asset_id, "image": image_url, "project_id": project_id, "panel_id": panel_id}


@router.get("/connection")
async def check_invokeai_connection():
    return await get_invokeai_client().check_connection()


@router.get("/models")
async def list_invokeai_models():
    try:
        records = await get_invokeai_client().get_model_records()
    except InvokeAIError as exc:
        logger.warning("InvokeAI model discovery failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    qwen_mains = [
        _public_model_record(record)
        for record in records
        if record.get("base") == "qwen-image" and record.get("type") == "main"
    ]
    qwen_loras = [
        _public_model_record(record)
        for record in records
        if record.get("base") == "qwen-image" and record.get("type") == "lora"
    ]
    vaes = [_public_model_record(record) for record in records if record.get("type") == "vae"]
    encoders = [
        _public_model_record(record)
        for record in records
        if record.get("type") == "qwen_vl_encoder"
    ]
    component_sources = [
        record for record in qwen_mains if record.get("format") == "diffusers"
    ]
    return {
        "models": qwen_mains,
        "loras": qwen_loras,
        "vae_models": vaes,
        "qwen_vl_encoders": encoders,
        "component_sources": component_sources,
    }


@router.post("/generate")
async def enqueue_invokeai_generation(req: InvokeAIGenerateRequest):
    _validate_project_panel(req.project_id, req.panel_id)
    client = get_invokeai_client()

    try:
        records = await client.get_model_records()
        main_model = _selected_record(records, req.model_key, label="Qwen Image main model")

        selected_loras = [
            {
                "model": _selected_record(records, item.model_key, label="LoRA"),
                "weight": item.weight,
            }
            for item in req.loras
        ]

        component_source = (
            _selected_record(records, req.component_source_key, label="component source")
            if req.component_source_key else None
        )
        vae_model = (
            _selected_record(records, req.vae_model_key, label="VAE")
            if req.vae_model_key else None
        )
        encoder_model = (
            _selected_record(records, req.qwen_vl_encoder_model_key, label="Qwen VL encoder")
            if req.qwen_vl_encoder_model_key else None
        )

        graph = build_qwen_image_2512_graph(
            main_model=main_model,
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            seed=req.seed,
            loras=selected_loras,
            component_source=component_source,
            vae_model=vae_model,
            qwen_vl_encoder_model=encoder_model,
            quantization=req.quantization,
            tiled_decode=req.tiled_decode,
        )
        destination = f"{_JOB_DESTINATION_PREFIX}{req.panel_id}"
        enqueue_result = await client.enqueue_graph(
            graph,
            origin="tt-automanga",
            project_id=req.project_id,
            destination=destination,
            idempotency_key=req.idempotency_key,
            queue_id=config.settings.invokeai.queue_id,
        )
        item_ids = enqueue_result["item_ids"]
        if len(item_ids) != 1:
            raise InvokeAIProtocolError(
                f"Expected one InvokeAI queue item for one generation; received {len(item_ids)}"
            )
        return {
            "success": True,
            "item_id": item_ids[0],
            "queue_id": config.settings.invokeai.queue_id,
            "batch_id": enqueue_result.get("batch_id"),
            "status": "pending",
            "project_id": req.project_id,
            "panel_id": req.panel_id,
        }
    except HTTPException:
        raise
    except QwenImageGraphError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvokeAIError as exc:
        logger.warning("InvokeAI generation enqueue failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/jobs/{item_id}")
async def get_invokeai_job_status(item_id: int = Path(gt=0)):
    client = get_invokeai_client()
    try:
        queue_item = await client.get_queue_item(
            item_id,
            queue_id=config.settings.invokeai.queue_id,
        )
        if queue_item.get("origin") != "tt-automanga":
            raise HTTPException(status_code=404, detail="TT AutoManga InvokeAI job not found")

        status = queue_item["status"]
        response: dict[str, Any] = {
            "item_id": item_id,
            "queue_id": config.settings.invokeai.queue_id,
            "status": status,
        }
        if status == "completed":
            persisted = await _persist_completed_job(client, queue_item, item_id)
            response.update(persisted)
        elif status in {"failed", "canceled"}:
            response["error"] = queue_item.get("error_message") or queue_item.get("error_type")
        return response
    except HTTPException:
        raise
    except (InvokeAIError, InvalidProjectIdError) as exc:
        logger.warning("InvokeAI job %s status/persistence failed: %s", item_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
