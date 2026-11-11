"""Contract tests for the native InvokeAI Qwen Image graph builder."""
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT.parent))

from backend.core.invokeai_qwen2512_graph import (
    DENOISE_NODE_ID,
    LORA_COLLECTION_NODE_ID,
    METADATA_NODE_ID,
    MODEL_LOADER_NODE_ID,
    NEGATIVE_ENCODER_NODE_ID,
    PROMPT_NODE_ID,
    QWEN_IMAGE_OUTPUT_NODE_ID,
    QwenImageGraphError,
    build_qwen_image_2512_graph,
)


MAIN_DIFFUSERS = {
    "key": "qwen-main-key",
    "hash": "main-hash",
    "name": "Qwen Image 2512",
    "base": "qwen-image",
    "type": "main",
    "format": "diffusers",
}
MAIN_GGUF = {
    **MAIN_DIFFUSERS,
    "key": "qwen-gguf-key",
    "format": "gguf_quantized",
}
COMPONENT_SOURCE = {
    **MAIN_DIFFUSERS,
    "key": "qwen-component-key",
    "name": "Qwen Image Components",
}
VAE = {
    "key": "qwen-vae-key",
    "hash": "vae-hash",
    "name": "Qwen Image VAE",
    "base": "qwen-image",
    "type": "vae",
}
ENCODER = {
    "key": "qwen-encoder-key",
    "hash": "encoder-hash",
    "name": "Qwen VL Encoder",
    "base": "any",
    "type": "qwen_vl_encoder",
}
LORA = {
    "key": "qwen-lora-key",
    "hash": "lora-hash",
    "name": "Character LoRA",
    "base": "qwen-image",
    "type": "lora",
}


def edge_tuple(edge):
    return (
        edge["source"]["node_id"],
        edge["source"]["field"],
        edge["destination"]["node_id"],
        edge["destination"]["field"],
    )


def test_builds_native_qwen_image_text_to_image_graph_with_metadata():
    graph = build_qwen_image_2512_graph(
        main_model=MAIN_DIFFUSERS,
        prompt="A manga panel, cinematic lighting",
        negative_prompt="blurry",
        graph_id="test-graph",
    )

    assert graph["id"] == "test-graph"
    nodes = graph["nodes"]
    assert nodes[MODEL_LOADER_NODE_ID]["type"] == "qwen_image_model_loader"
    assert nodes[PROMPT_NODE_ID]["type"] == "qwen_image_text_encoder"
    assert nodes[DENOISE_NODE_ID]["type"] == "qwen_image_denoise"
    assert nodes[QWEN_IMAGE_OUTPUT_NODE_ID]["type"] == "qwen_image_l2i"
    assert nodes[QWEN_IMAGE_OUTPUT_NODE_ID]["is_intermediate"] is False
    assert nodes[METADATA_NODE_ID]["generation_mode"] == "qwen_image_txt2img"
    assert nodes[METADATA_NODE_ID]["positive_prompt"] == "A manga panel, cinematic lighting"
    assert nodes[METADATA_NODE_ID]["negative_prompt"] == "blurry"


def test_base_graph_connects_all_required_qwen_components_and_metadata():
    graph = build_qwen_image_2512_graph(
        main_model=MAIN_DIFFUSERS,
        prompt="manga",
        cfg_scale=1.0,
    )
    edges = {edge_tuple(edge) for edge in graph["edges"]}

    assert edges == {
        (MODEL_LOADER_NODE_ID, "transformer", DENOISE_NODE_ID, "transformer"),
        (MODEL_LOADER_NODE_ID, "qwen_vl_encoder", PROMPT_NODE_ID, "qwen_vl_encoder"),
        (PROMPT_NODE_ID, "conditioning", DENOISE_NODE_ID, "positive_conditioning"),
        (DENOISE_NODE_ID, "latents", QWEN_IMAGE_OUTPUT_NODE_ID, "latents"),
        (MODEL_LOADER_NODE_ID, "vae", QWEN_IMAGE_OUTPUT_NODE_ID, "vae"),
        (METADATA_NODE_ID, "metadata", QWEN_IMAGE_OUTPUT_NODE_ID, "metadata"),
    }


def test_cfg_above_one_wires_negative_conditioning():
    graph = build_qwen_image_2512_graph(
        main_model=MAIN_DIFFUSERS,
        prompt="manga",
        negative_prompt="blurry",
        cfg_scale=4.0,
    )
    nodes = graph["nodes"]
    edges = {edge_tuple(edge) for edge in graph["edges"]}
    assert nodes[NEGATIVE_ENCODER_NODE_ID]["prompt"] == "blurry"
    assert (
        NEGATIVE_ENCODER_NODE_ID, "conditioning", DENOISE_NODE_ID, "negative_conditioning"
    ) in edges


def test_cfg_at_or_below_one_omits_negative_conditioning_node():
    graph = build_qwen_image_2512_graph(
        main_model=MAIN_DIFFUSERS,
        prompt="manga",
        negative_prompt="blurry",
        cfg_scale=1.0,
    )
    assert NEGATIVE_ENCODER_NODE_ID not in graph["nodes"]
    assert not any(edge["destination"]["field"] == "negative_conditioning" for edge in graph["edges"])
    assert graph["nodes"][METADATA_NODE_ID]["negative_prompt"] == "blurry"


def test_empty_negative_prompt_uses_nonempty_encoder_prompt_when_cfg_enabled():
    graph = build_qwen_image_2512_graph(main_model=MAIN_DIFFUSERS, prompt="manga", cfg_scale=4.0)
    assert graph["nodes"][NEGATIVE_ENCODER_NODE_ID]["prompt"] == " "


def test_graph_supports_multiple_loras_and_routes_both_model_streams():
    graph = build_qwen_image_2512_graph(
        main_model=MAIN_DIFFUSERS,
        prompt="manga",
        loras=[
            {"model": LORA, "weight": 0.8},
            {"model": {**LORA, "key": "second-lora", "name": "Style LoRA"}, "weight": 0.55},
        ],
    )

    lora_node = graph["nodes"][LORA_COLLECTION_NODE_ID]
    assert lora_node["type"] == "qwen_image_lora_collection_loader"
    assert [item["lora"]["key"] for item in lora_node["loras"]] == [
        "qwen-lora-key",
        "second-lora",
    ]
    edges = {edge_tuple(edge) for edge in graph["edges"]}
    assert (MODEL_LOADER_NODE_ID, "transformer", LORA_COLLECTION_NODE_ID, "transformer") in edges
    assert (MODEL_LOADER_NODE_ID, "qwen_vl_encoder", LORA_COLLECTION_NODE_ID, "qwen_vl_encoder") in edges
    assert (LORA_COLLECTION_NODE_ID, "transformer", DENOISE_NODE_ID, "transformer") in edges
    assert (LORA_COLLECTION_NODE_ID, "qwen_vl_encoder", PROMPT_NODE_ID, "qwen_vl_encoder") in edges
    assert (
        LORA_COLLECTION_NODE_ID, "qwen_vl_encoder", NEGATIVE_ENCODER_NODE_ID, "qwen_vl_encoder"
    ) in edges
    assert graph["nodes"][METADATA_NODE_ID]["loras"][0]["model"]["key"] == "qwen-lora-key"


def test_graph_without_loras_does_not_add_lora_node():
    graph = build_qwen_image_2512_graph(main_model=MAIN_DIFFUSERS, prompt="manga", loras=[])
    assert LORA_COLLECTION_NODE_ID not in graph["nodes"]


def test_gguf_main_requires_component_source_or_both_standalone_components():
    with pytest.raises(QwenImageGraphError, match="non-Diffusers"):
        build_qwen_image_2512_graph(main_model=MAIN_GGUF, prompt="manga")

    with pytest.raises(QwenImageGraphError, match="non-Diffusers"):
        build_qwen_image_2512_graph(main_model=MAIN_GGUF, prompt="manga", vae_model=VAE)

    graph = build_qwen_image_2512_graph(
        main_model=MAIN_GGUF,
        prompt="manga",
        vae_model=VAE,
        qwen_vl_encoder_model=ENCODER,
    )
    node = graph["nodes"][MODEL_LOADER_NODE_ID]
    assert node["vae_model"]["key"] == VAE["key"]
    assert node["qwen_vl_encoder_model"]["key"] == ENCODER["key"]


def test_gguf_main_accepts_diffusers_component_source():
    graph = build_qwen_image_2512_graph(
        main_model=MAIN_GGUF,
        prompt="manga",
        component_source=COMPONENT_SOURCE,
    )
    assert graph["nodes"][MODEL_LOADER_NODE_ID]["component_source"]["key"] == COMPONENT_SOURCE["key"]


def test_component_source_must_be_diffusers_format():
    invalid = {**COMPONENT_SOURCE, "format": "gguf_quantized"}
    with pytest.raises(QwenImageGraphError, match="Diffusers-format"):
        build_qwen_image_2512_graph(
            main_model=MAIN_GGUF,
            prompt="manga",
            component_source=invalid,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": ""},
        {"width": 1000},
        {"height": 1001},
        {"steps": 0},
        {"cfg_scale": float("nan")},
        {"seed": True},
        {"quantization": "fp8"},
        {"tiled_decode": "false"},
    ],
)
def test_invalid_generation_options_are_rejected(kwargs):
    options = {"prompt": "manga"}
    options.update(kwargs)
    with pytest.raises(QwenImageGraphError):
        build_qwen_image_2512_graph(main_model=MAIN_DIFFUSERS, **options)


def test_model_records_must_match_expected_architecture_and_type():
    with pytest.raises(QwenImageGraphError, match="base"):
        build_qwen_image_2512_graph(
            main_model={**MAIN_DIFFUSERS, "base": "sdxl"},
            prompt="manga",
        )

    with pytest.raises(QwenImageGraphError, match="type"):
        build_qwen_image_2512_graph(
            main_model={**MAIN_DIFFUSERS, "type": "lora"},
            prompt="manga",
        )


def test_loras_must_be_qwen_image_loras_with_unique_keys():
    with pytest.raises(QwenImageGraphError, match="base"):
        build_qwen_image_2512_graph(
            main_model=MAIN_DIFFUSERS,
            prompt="manga",
            loras=[{"model": {**LORA, "base": "sdxl"}, "weight": 1.0}],
        )

    with pytest.raises(QwenImageGraphError, match="more than once"):
        build_qwen_image_2512_graph(
            main_model=MAIN_DIFFUSERS,
            prompt="manga",
            loras=[
                {"model": LORA, "weight": 0.8},
                {"model": LORA, "weight": 0.5},
            ],
        )


def test_lora_weights_must_be_finite_numbers():
    with pytest.raises(QwenImageGraphError, match="finite"):
        build_qwen_image_2512_graph(
            main_model=MAIN_DIFFUSERS,
            prompt="manga",
            loras=[{"model": LORA, "weight": float("inf")}],
        )


def test_graph_has_invokeai_node_flags_and_valid_output_node_id():
    graph = build_qwen_image_2512_graph(main_model=MAIN_DIFFUSERS, prompt="manga")
    assert graph["nodes"][MODEL_LOADER_NODE_ID]["is_intermediate"] is True
    assert graph["nodes"][MODEL_LOADER_NODE_ID]["use_cache"] is True
    assert graph["nodes"][QWEN_IMAGE_OUTPUT_NODE_ID]["is_intermediate"] is False
    assert QWEN_IMAGE_OUTPUT_NODE_ID in graph["nodes"]
