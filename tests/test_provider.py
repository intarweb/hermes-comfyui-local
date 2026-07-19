"""Provider-level tests: config resolution (host used verbatim), model selection,
and the generate() orchestration end-to-end with ComfyUI HTTP mocked. The Hermes
runtime is stubbed in conftest.py."""

from __future__ import annotations

import json
import sys

import pytest
import responses

_HOST = "http://comfy.test:8188"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("COMFYUI_HOST", "COMFY_HOST", "COMFYUI_API_KEY",
                "COMFY_CLOUD_API_KEY", "COMFYUI_IMAGE_WORKFLOW"):
        monkeypatch.delenv(var, raising=False)
    # default: no config
    monkeypatch.setattr(sys.modules["hermes_cli.config"], "load_config", lambda: {})


# --- config resolution ---------------------------------------------------- #
def test_host_from_env_verbatim(plugin, monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", "http://box:9999/")
    assert plugin._comfy_host() == "http://box:9999"   # only trailing slash trimmed


def test_host_from_config_preserves_port(plugin, monkeypatch):
    # fix #3: the configured port must be used as-is, never "discovered"/rewritten.
    monkeypatch.setattr(
        sys.modules["hermes_cli.config"], "load_config",
        lambda: {"image_gen": {"comfyui": {"host": "http://comfy-box:18188"}}},
    )
    assert plugin._comfy_host() == "http://comfy-box:18188"


def test_host_default(plugin):
    assert plugin._comfy_host() == "http://127.0.0.1:8188"


def test_env_overrides_config(plugin, monkeypatch):
    monkeypatch.setattr(
        sys.modules["hermes_cli.config"], "load_config",
        lambda: {"image_gen": {"comfyui": {"host": "http://from-config:8188"}}},
    )
    monkeypatch.setenv("COMFYUI_HOST", "http://from-env:8188")
    assert plugin._comfy_host() == "http://from-env:8188"


# --- model selection ------------------------------------------------------ #
def test_default_model_is_flux(plugin):
    assert plugin.ComfyUIImageGenProvider().default_model() == "flux_dev_txt2img"


def test_list_models_has_all_bundled(plugin):
    ids = {m["id"] for m in plugin.ComfyUIImageGenProvider().list_models()}
    assert {"flux_dev_txt2img", "sdxl_txt2img", "sd15_txt2img"} <= ids


def test_resolve_model_requested_wins(plugin):
    model_id, _ = plugin._resolve_model("sd15_txt2img")
    assert model_id == "sd15_txt2img"


# --- generate() end-to-end ------------------------------------------------ #
def test_generate_validation_error_on_empty_prompt(plugin):
    res = plugin.ComfyUIImageGenProvider().generate("   ", "square")
    assert res["success"] is False and res["error_type"] == "validation_error"


@responses.activate
def test_generate_backend_unavailable(plugin, monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", _HOST)
    responses.add(responses.GET, f"{_HOST}/system_stats", status=500)
    res = plugin.ComfyUIImageGenProvider().generate("a cat", "square")
    assert res["success"] is False and res["error_type"] == "backend_unavailable"


@responses.activate
def test_generate_happy_path_returns_local_media_path(plugin, monkeypatch, cache_dir):
    monkeypatch.setenv("COMFYUI_HOST", _HOST)
    monkeypatch.setenv("COMFYUI_IMAGE_WORKFLOW", "sdxl_txt2img")  # deterministic pick

    responses.add(responses.GET, f"{_HOST}/system_stats", json={"ok": 1}, status=200)
    responses.add(responses.POST, f"{_HOST}/prompt", json={"prompt_id": "abc12345"}, status=200)
    responses.add(
        responses.GET, f"{_HOST}/history/abc12345",
        json={"abc12345": {
            "status": {"completed": True},
            "outputs": {"9": {"images": [{"filename": "sdxl_0001.png", "subfolder": "", "type": "output"}]}},
        }},
        status=200,
    )
    responses.add(responses.GET, f"{_HOST}/view", body=b"\x89PNG-bytes", status=200)

    res = plugin.ComfyUIImageGenProvider().generate("a red fox", "landscape", seed=7)

    assert res["success"] is True
    assert res["provider"] == "comfyui"
    assert res["model"] == "sdxl_txt2img"
    # image is an ABSOLUTE local path (MEDIA: delivery), NOT markdown
    img = res["image"]
    assert img.startswith(str(cache_dir))
    assert img.endswith(".png") and "![" not in img and "](" not in img
    # bytes actually written
    from pathlib import Path
    assert Path(img).read_bytes() == b"\x89PNG-bytes"
    # host echoed verbatim, size mapped from aspect
    assert res["extra"]["host"] == _HOST
    assert res["extra"]["width"] == 1024 and res["extra"]["height"] == 768


@responses.activate
def test_generate_api_error_on_comfy_failure(plugin, monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", _HOST)
    responses.add(responses.GET, f"{_HOST}/system_stats", json={"ok": 1}, status=200)
    responses.add(responses.POST, f"{_HOST}/prompt", body="boom", status=500)
    res = plugin.ComfyUIImageGenProvider().generate("a cat", "square")
    assert res["success"] is False and res["error_type"] == "api_error"


# --- register() ----------------------------------------------------------- #
def test_register_registers_one_provider(plugin):
    registered = []

    class Ctx:
        def register_image_gen_provider(self, provider):
            registered.append(provider)

    plugin.register(Ctx())
    assert len(registered) == 1
    assert registered[0].name == "comfyui"
