"""Build InvokeAI's native Qwen Image text-to-image graph.

The node names, ports and model identifier payloads mirror InvokeAI's built-in
Qwen Image invocations. This module does not call the server; it produces the
Graph object accepted by InvokeAIClient.enqueue_graph().
"""
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from typing import Any


QWEN_IMAGE_BASE = "qwen-image"
QWEN_IMAGE_OUTPUT_NODE_ID = "qwen_decode"

MODEL_LOADER_NODE_ID = "qwen_model_loader"
PROMPT_NODE_ID = "qwen_prompt"
LORA_COLLECTION_NODE_ID = "qwen_lora_collection"
DENOISE_NODE_ID = "qwen_denoise"
NEGATIVE_ENCODER_NODE_ID = "qwen_negative_prompt"
METADATA_NODE_ID = "qwen_core_metadata"

_NODE_FLAGS = {"is_intermediate": True, "use_cache": True}


class QwenImageGraphError(ValueError):
    """Invalid model selection or generation options for a Qwen Image graph."""


def _required_text(record: Mapping[str, Any], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise QwenImageGraphError(f"{label} is missing a valid {field!r}")
    return value.strip()


def _model_identifier(
    record: Mapping[str, Any],
    *,
    label: str,
    expected_type: str,
    expected_base: str | None = None,
) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise QwenImageGraphError(f"{label} must be an InvokeAI model record")

    identifier = {
        field: _required_text(record, field, label)
        for field in ("key", "hash", "name", "base", "type")
    }
    if identifier["type"] != expected_type:
        raise QwenImageGraphError(
            f"{label} must have type {expected_type!r}; got {identifier['type']!r}"
        )
    if expected_base is not None and identifier["base"] != expected_base:
        raise QwenImageGraphError(
            f"{label} must have base {expected_base!r}; got {identifier['base']!r}"
        )
    return identifier


def _edge(
    source_node: str,
    source_field: str,
    destination_node: str,
    destination_field: str,
) -> dict[str, dict[str, str]]:
    return {
        "source": {"node_id": source_node, "field": source_field},
        "destination": {"node_id": destination_node, "field": destination_field},
    }


def _validate_generation_options(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    quantization: str,
    tiled_decode: bool,
) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise QwenImageGraphError("prompt must be a non-empty string")
    if not isinstance(negative_prompt, str):
        raise QwenImageGraphError("negative_prompt must be a string")

    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 16 or value % 16:
            raise QwenImageGraphError(f"{name} must be a positive integer multiple of 16")

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise QwenImageGraphError("steps must be a positive integer")

    if isinstance(cfg_scale, bool) or not isinstance(cfg_scale, (int, float)) or not math.isfinite(cfg_scale):
        raise QwenImageGraphError("cfg_scale must be a finite number")
    if cfg_scale < 0:
        raise QwenImageGraphError("cfg_scale cannot be negative")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise QwenImageGraphError("seed must be an integer")

    if quantization not in {"none", "int8", "nf4"}:
        raise QwenImageGraphError("quantization must be one of: none, int8, nf4")
    if not isinstance(tiled_decode, bool):
        raise QwenImageGraphError("tiled_decode must be a boolean")


def build_qwen_image_2512_graph(
    *,
    main_model: Mapping[str, Any],
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 40,
    cfg_scale: float = 4.0,
    seed: int = 0,
    loras: Sequence[Mapping[str, Any]] | None = None,
    component_source: Mapping[str, Any] | None = None,
    vae_model: Mapping[str, Any] | None = None,
    qwen_vl_encoder_model: Mapping[str, Any] | None = None,
    quantization: str = "none",
    tiled_decode: bool = False,
    graph_id: str | None = None,
) -> dict[str, Any]:
    """Build a Qwen Image 2512 text-to-image graph using InvokeAI's native nodes.

    main_model, optional component records, and each LoRA model value are the
    model records returned by GET /api/v2/models/. Each LoRA entry is
    {"model": <model record>, "weight": <float>}.
    """
    _validate_generation_options(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        quantization=quantization,
        tiled_decode=tiled_decode,
    )

    main_identifier = _model_identifier(
        main_model,
        label="main_model",
        expected_type="main",
        expected_base=QWEN_IMAGE_BASE,
    )
    main_format = main_model.get("format")
    if not isinstance(main_format, str) or not main_format.strip():
        raise QwenImageGraphError("main_model is missing its format metadata")
    main_format = main_format.lower()

    component_identifier = None
    if component_source is not None:
        component_identifier = _model_identifier(
            component_source,
            label="component_source",
            expected_type="main",
            expected_base=QWEN_IMAGE_BASE,
        )
        if str(component_source.get("format", "")).lower() != "diffusers":
            raise QwenImageGraphError("component_source must be a Diffusers-format Qwen Image main model")

    vae_identifier = None
    if vae_model is not None:
        vae_identifier = _model_identifier(
            vae_model,
            label="vae_model",
            expected_type="vae",
        )

    encoder_identifier = None
    if qwen_vl_encoder_model is not None:
        encoder_identifier = _model_identifier(
            qwen_vl_encoder_model,
            label="qwen_vl_encoder_model",
            expected_type="qwen_vl_encoder",
        )

    if main_format != "diffusers":
        has_vae_source = vae_identifier is not None or component_identifier is not None
        has_encoder_source = encoder_identifier is not None or component_identifier is not None
        if not has_vae_source or not has_encoder_source:
            raise QwenImageGraphError(
                "A non-Diffusers main model requires a Diffusers component_source, "
                "or both vae_model and qwen_vl_encoder_model"
            )

    normalized_loras: list[dict[str, Any]] = []
    seen_lora_keys: set[str] = set()
    for index, entry in enumerate(loras or ()):
        if not isinstance(entry, Mapping):
            raise QwenImageGraphError(f"loras[{index}] must be an object")
        lora_record = entry.get("model")
        lora_identifier = _model_identifier(
            lora_record,
            label=f"loras[{index}].model",
            expected_type="lora",
            expected_base=QWEN_IMAGE_BASE,
        )
        if lora_identifier["key"] in seen_lora_keys:
            raise QwenImageGraphError(f"LoRA {lora_identifier['key']!r} is selected more than once")
        seen_lora_keys.add(lora_identifier["key"])

        weight = entry.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise QwenImageGraphError(f"loras[{index}].weight must be a finite number")
        normalized_loras.append({"lora": lora_identifier, "weight": float(weight)})

    if graph_id is None:
        graph_id = uuid.uuid4().hex
    elif not isinstance(graph_id, str) or not graph_id.strip():
        raise QwenImageGraphError("graph_id must be a non-empty string")

    model_node: dict[str, Any] = {
        "id": MODEL_LOADER_NODE_ID,
        "type": "qwen_image_model_loader",
        **_NODE_FLAGS,
        "model": main_identifier,
    }
    if vae_identifier is not None:
        model_node["vae_model"] = vae_identifier
    if encoder_identifier is not None:
        model_node["qwen_vl_encoder_model"] = encoder_identifier
    if component_identifier is not None:
        model_node["component_source"] = component_identifier

    prompt_node = {
        "id": PROMPT_NODE_ID,
        "type": "qwen_image_text_encoder",
        **_NODE_FLAGS,
        "prompt": prompt.strip(),
        "reference_images": [],
        "quantization": quantization,
    }
    denoise_node = {
        "id": DENOISE_NODE_ID,
        "type": "qwen_image_denoise",
        **_NODE_FLAGS,
        "cfg_scale": float(cfg_scale),
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
    }
    decode_node = {
        "id": QWEN_IMAGE_OUTPUT_NODE_ID,
        "type": "qwen_image_l2i",
        **_NODE_FLAGS,
        "is_intermediate": False,
        "tiled": tiled_decode,
        "tile_size": 0,
    }
    metadata_node: dict[str, Any] = {
        "id": METADATA_NODE_ID,
        "type": "core_metadata",
        **_NODE_FLAGS,
        "generation_mode": "qwen_image_txt2img",
        "positive_prompt": prompt.strip(),
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "cfg_scale": float(cfg_scale),
        "steps": steps,
        "model": main_identifier,
        "loras": [
            {"model": item["lora"], "weight": item["weight"]}
            for item in normalized_loras
        ] or None,
    }
    if vae_identifier is not None:
        metadata_node["vae"] = vae_identifier

    nodes: dict[str, dict[str, Any]] = {
        MODEL_LOADER_NODE_ID: model_node,
        PROMPT_NODE_ID: prompt_node,
        DENOISE_NODE_ID: denoise_node,
        QWEN_IMAGE_OUTPUT_NODE_ID: decode_node,
        METADATA_NODE_ID: metadata_node,
    }

    edges: list[dict[str, dict[str, str]]] = [
        _edge(DENOISE_NODE_ID, "latents", QWEN_IMAGE_OUTPUT_NODE_ID, "latents"),
        _edge(MODEL_LOADER_NODE_ID, "vae", QWEN_IMAGE_OUTPUT_NODE_ID, "vae"),
        _edge(METADATA_NODE_ID, "metadata", QWEN_IMAGE_OUTPUT_NODE_ID, "metadata"),
    ]

    if normalized_loras:
        lora_node = {
            "id": LORA_COLLECTION_NODE_ID,
            "type": "qwen_image_lora_collection_loader",
            **_NODE_FLAGS,
            "loras": normalized_loras,
        }
        nodes[LORA_COLLECTION_NODE_ID] = lora_node
        edges.extend([
            _edge(MODEL_LOADER_NODE_ID, "transformer", LORA_COLLECTION_NODE_ID, "transformer"),
            _edge(MODEL_LOADER_NODE_ID, "qwen_vl_encoder", LORA_COLLECTION_NODE_ID, "qwen_vl_encoder"),
            _edge(LORA_COLLECTION_NODE_ID, "transformer", DENOISE_NODE_ID, "transformer"),
            _edge(LORA_COLLECTION_NODE_ID, "qwen_vl_encoder", PROMPT_NODE_ID, "qwen_vl_encoder"),
        ])
    else:
        edges.extend([
            _edge(MODEL_LOADER_NODE_ID, "transformer", DENOISE_NODE_ID, "transformer"),
            _edge(MODEL_LOADER_NODE_ID, "qwen_vl_encoder", PROMPT_NODE_ID, "qwen_vl_encoder"),
        ])

    if float(cfg_scale) > 1:
        negative_node = {
            "id": NEGATIVE_ENCODER_NODE_ID,
            "type": "qwen_image_text_encoder",
            **_NODE_FLAGS,
            "prompt": negative_prompt or " ",
            "reference_images": [],
            "quantization": quantization,
        }
        nodes[NEGATIVE_ENCODER_NODE_ID] = negative_node
        negative_encoder_source = (
            LORA_COLLECTION_NODE_ID if normalized_loras else MODEL_LOADER_NODE_ID
        )
        edges.extend([
            _edge(
                negative_encoder_source,
                "qwen_vl_encoder",
                NEGATIVE_ENCODER_NODE_ID,
                "qwen_vl_encoder",
            ),
            _edge(
                NEGATIVE_ENCODER_NODE_ID,
                "conditioning",
                DENOISE_NODE_ID,
                "negative_conditioning",
            ),
        ])

    edges.append(
        _edge(PROMPT_NODE_ID, "conditioning", DENOISE_NODE_ID, "positive_conditioning")
    )

    return {"id": graph_id, "nodes": nodes, "edges": edges}
