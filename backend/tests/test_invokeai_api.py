"""API contract tests for InvokeAI generation orchestration."""
import io
import sys
from copy import deepcopy
from pathlib import Path

from PIL import Image
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT.parent))

from backend.main import app
from backend.storage.asset_store import AssetStore


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (3, 3), (20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


MAIN = {
    "key": "qwen-main",
    "hash": "main-hash",
    "name": "Qwen Image 2512",
    "base": "qwen-image",
    "type": "main",
    "format": "diffusers",
}
LORA = {
    "key": "qwen-lora",
    "hash": "lora-hash",
    "name": "Character LoRA",
    "base": "qwen-image",
    "type": "lora",
}
VAE = {
    "key": "qwen-vae",
    "hash": "vae-hash",
    "name": "Qwen VAE",
    "base": "qwen-image",
    "type": "vae",
}
ENCODER = {
    "key": "qwen-encoder",
    "hash": "encoder-hash",
    "name": "Qwen VL Encoder",
    "base": "any",
    "type": "qwen_vl_encoder",
}
SDXL = {
    "key": "sdxl-main",
    "hash": "sdxl-hash",
    "name": "SDXL",
    "base": "sdxl",
    "type": "main",
    "format": "diffusers",
}


class FakeStorage:
    def __init__(self):
        self.project = {
            "id": "project1",
            "name": "Test Project",
            "panels": [{"id": "panel1", "scene": "A test scene", "image": None}],
            "settings": {},
            "composerPages": [],
        }

    def load_project(self, project_id):
        if project_id != self.project["id"]:
            return None
        return deepcopy(self.project)

    def save_project(self, project):
        self.project = deepcopy(project)
        return project


class FakeInvokeAIClient:
    def __init__(self):
        self.records = [MAIN, LORA, VAE, ENCODER, SDXL]
        self.enqueued = None
        self.queue_item = {
            "item_id": 73,
            "queue_id": "default",
            "origin": "tt-automanga",
            "destination": "tt-automanga:panel:panel1",
            "project_id": "project1",
            "status": "completed",
            "session": {
                "results": {
                    "qwen_decode": {
                        "image": {"image_name": "generated-qwen.png"}
                    }
                }
            },
        }
        self.downloads = 0

    async def check_connection(self):
        return {"connected": True, "status": "ok", "version": "test"}

    async def get_model_records(self):
        return self.records

    async def enqueue_graph(self, graph, **kwargs):
        self.enqueued = {"graph": graph, **kwargs}
        return {"item_ids": [73], "batch_id": "batch-73"}

    async def get_queue_item(self, item_id, queue_id="default"):
        assert item_id == 73
        return deepcopy(self.queue_item)

    async def download_image(self, image_name):
        assert image_name == "generated-qwen.png"
        self.downloads += 1
        return png_bytes()


def setup_api(monkeypatch, tmp_path):
    from backend.api import invokeai as api

    fake_client = FakeInvokeAIClient()
    fake_storage = FakeStorage()
    fake_assets = AssetStore(tmp_path / "assets", tmp_path / "assets" / "index.json")
    monkeypatch.setattr(api, "get_invokeai_client", lambda: fake_client)
    monkeypatch.setattr(api, "storage", fake_storage)
    monkeypatch.setattr(api, "asset_store", fake_assets)
    return TestClient(app), fake_client, fake_storage, fake_assets


def test_model_discovery_returns_only_selection_metadata(monkeypatch, tmp_path):
    client, _, _, _ = setup_api(monkeypatch, tmp_path)
    response = client.get("/api/invokeai/models")

    assert response.status_code == 200, response.text
    data = response.json()
    assert [item["key"] for item in data["models"]] == ["qwen-main"]
    assert [item["key"] for item in data["loras"]] == ["qwen-lora"]
    assert [item["key"] for item in data["vae_models"]] == ["qwen-vae"]
    assert [item["key"] for item in data["qwen_vl_encoders"]] == ["qwen-encoder"]
    assert "path" not in data["models"][0]


def test_generate_enqueues_native_graph_with_panel_destination(monkeypatch, tmp_path):
    client, fake_client, _, _ = setup_api(monkeypatch, tmp_path)
    response = client.post(
        "/api/invokeai/generate",
        json={
            "project_id": "project1",
            "panel_id": "panel1",
            "prompt": "A cinematic manga panel",
            "model_key": "qwen-main",
            "loras": [{"model_key": "qwen-lora", "weight": 0.75}],
            "width": 1024,
            "height": 1024,
            "steps": 40,
            "cfg_scale": 4.0,
            "seed": 123,
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["item_id"] == 73
    assert data["status"] == "pending"
    assert fake_client.enqueued["origin"] == "tt-automanga"
    assert fake_client.enqueued["project_id"] == "project1"
    assert fake_client.enqueued["destination"] == "tt-automanga:panel:panel1"
    graph = fake_client.enqueued["graph"]
    assert graph["nodes"]["qwen_lora_collection"]["type"] == "qwen_image_lora_collection_loader"


def test_completed_job_persists_panel_asset_once(monkeypatch, tmp_path):
    client, fake_client, fake_storage, fake_assets = setup_api(monkeypatch, tmp_path)

    first = client.get("/api/invokeai/jobs/73")
    second = client.get("/api/invokeai/jobs/73")

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_data = first.json()
    second_data = second.json()
    assert first_data["status"] == "completed"
    assert first_data["asset_id"] == second_data["asset_id"]
    assert first_data["image"] == second_data["image"]
    assert fake_client.downloads == 1
    assert len(fake_assets.ids()) == 1
    assert fake_storage.project["panels"][0]["image"] == first_data["image"]


def test_failed_job_does_not_create_or_attach_asset(monkeypatch, tmp_path):
    client, fake_client, fake_storage, fake_assets = setup_api(monkeypatch, tmp_path)
    fake_client.queue_item["status"] = "failed"
    fake_client.queue_item["error_message"] = "CUDA out of memory"

    response = client.get("/api/invokeai/jobs/73")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "failed"
    assert response.json()["error"] == "CUDA out of memory"
    assert len(fake_assets.ids()) == 0
    assert fake_storage.project["panels"][0]["image"] is None


def test_job_from_another_origin_is_not_exposed(monkeypatch, tmp_path):
    client, fake_client, _, _ = setup_api(monkeypatch, tmp_path)
    fake_client.queue_item["origin"] = "other-app"

    response = client.get("/api/invokeai/jobs/73")

    assert response.status_code == 404
