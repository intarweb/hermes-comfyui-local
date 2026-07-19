"""Unit tests for comfyui_client: the parts with no Hermes-runtime dependency —
the /prompt -> /history -> /view HTTP dance, non-node stripping, workflow spec
inference, field injection, and aspect sizing. All HTTP is mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import responses

import comfyui_client as cc

_WF_DIR = Path(__file__).resolve().parent.parent / "workflows"
_HOST = "http://comfy.test:8188"


def _load(name):
    return json.loads((_WF_DIR / name).read_text())


# --- strip_non_nodes (fix #1) --------------------------------------------- #
def test_strip_non_nodes_drops_annotations():
    wf = {
        "_comment": "author note",
        "note": "not a node either",
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
    }
    stripped = cc.strip_non_nodes(wf)
    assert set(stripped) == {"6"}
    assert "_comment" not in stripped and "note" not in stripped


def test_strip_non_nodes_keeps_all_real_nodes():
    wf = _load("sdxl_txt2img.json")
    stripped = cc.strip_non_nodes(wf)
    # bundled workflows ship comment-free, so nothing is dropped
    assert stripped == wf
    assert all("class_type" in node for node in stripped.values())


# --- infer_workflow_spec -------------------------------------------------- #
def test_infer_spec_sdxl():
    spec = cc.infer_workflow_spec(_load("sdxl_txt2img.json"))
    assert spec["prompt"] == ("6", "text")        # positive
    assert spec["negative"] == ("7", "text")       # negative by title
    assert spec["latent"] == ("5", "width", "height")
    assert ("3", "seed") in spec["seed_fields"]    # KSampler
    assert spec["base_size"] == 1024


def test_infer_spec_flux_uses_randomnoise_and_sd3_latent():
    spec = cc.infer_workflow_spec(_load("flux_dev_txt2img.json"))
    assert spec["prompt"] == ("6", "text")
    assert spec["latent"] == ("27", "width", "height")   # EmptySD3LatentImage
    assert ("25", "noise_seed") in spec["seed_fields"]   # RandomNoise
    assert spec["negative"] is None                      # flux graph has no negative


# --- inject_workflow ------------------------------------------------------ #
def test_inject_sets_prompt_size_and_seed():
    wf = _load("sdxl_txt2img.json")
    spec = cc.infer_workflow_spec(wf)
    out = cc.inject_workflow(
        wf, prompt="a red fox", width=768, height=1024, seed=99,
        spec=spec, negative_prompt="blurry",
    )
    assert out["6"]["inputs"]["text"] == "a red fox"
    assert out["7"]["inputs"]["text"] == "blurry"
    assert out["5"]["inputs"]["width"] == 768 and out["5"]["inputs"]["height"] == 1024
    assert out["3"]["inputs"]["seed"] == 99
    # original workflow not mutated
    assert wf["6"]["inputs"]["text"] != "a red fox"


def test_inject_raises_without_prompt_node():
    spec = {"prompt": None, "negative": None, "latent": None, "seed_fields": [], "base_size": 1024}
    with pytest.raises(cc.ComfyUIError):
        cc.inject_workflow({}, prompt="x", width=512, height=512, spec=spec)


def test_inject_skips_linked_seed_input():
    # A seed field wired from another node ([node, slot]) must not be overwritten.
    wf = {
        "1": {"class_type": "KSampler", "inputs": {"seed": ["2", 0], "text": None}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    }
    spec = {"prompt": ("9", "text"), "negative": None, "latent": None,
            "seed_fields": [("1", "seed")], "base_size": 1024}
    out = cc.inject_workflow(wf, prompt="hi", width=512, height=512, seed=7, spec=spec)
    assert out["1"]["inputs"]["seed"] == ["2", 0]   # left as a link


# --- aspect_to_size ------------------------------------------------------- #
@pytest.mark.parametrize("aspect,expected", [
    ("square", (1024, 1024)),
    ("landscape", (1024, 768)),
    ("portrait", (768, 1024)),
    ("bogus", (1024, 1024)),   # else-branch -> square
])
def test_aspect_to_size(aspect, expected):
    assert cc.aspect_to_size(aspect, base=1024) == expected


def test_aspect_to_size_snaps_to_multiple_of_8():
    w, h = cc.aspect_to_size("landscape", base=1000)   # 1000, 750 -> snap
    assert w % 8 == 0 and h % 8 == 0 and w >= 256 and h >= 256


# --- ComfyUIClient HTTP dance --------------------------------------------- #
@responses.activate
def test_submit_strips_non_nodes_before_post():
    captured = {}

    def _cb(request):
        captured["body"] = json.loads(request.body)
        return (200, {}, json.dumps({"prompt_id": "pid-1"}))

    responses.add_callback(responses.POST, f"{_HOST}/prompt", callback=_cb)
    client = cc.ComfyUIClient(_HOST)
    pid = client.submit({"_comment": "x", "6": {"class_type": "CLIPTextEncode", "inputs": {}}})
    assert pid == "pid-1"
    assert "_comment" not in captured["body"]["prompt"]   # stripped
    assert "6" in captured["body"]["prompt"]
    assert captured["body"]["client_id"] == client.client_id


@responses.activate
def test_submit_raises_on_node_errors():
    responses.add(responses.POST, f"{_HOST}/prompt",
                  json={"prompt_id": "p", "node_errors": {"6": "bad"}}, status=200)
    with pytest.raises(cc.ComfyUIError):
        cc.ComfyUIClient(_HOST).submit({"6": {"class_type": "X"}})


@responses.activate
def test_submit_raises_on_missing_prompt_id():
    responses.add(responses.POST, f"{_HOST}/prompt", json={}, status=200)
    with pytest.raises(cc.ComfyUIError):
        cc.ComfyUIClient(_HOST).submit({"6": {"class_type": "X"}})


@responses.activate
def test_poll_returns_entry_on_completed():
    responses.add(responses.GET, f"{_HOST}/history/pid-1",
                  json={"pid-1": {"status": {"completed": True}, "outputs": {}}}, status=200)
    entry = cc.ComfyUIClient(_HOST).poll("pid-1", poll_interval=0.01)
    assert entry["status"]["completed"] is True


@responses.activate
def test_poll_raises_on_execution_error():
    responses.add(responses.GET, f"{_HOST}/history/pid-2",
                  json={"pid-2": {"status": {"status_str": "error"}}}, status=200)
    with pytest.raises(cc.ComfyUIError):
        cc.ComfyUIClient(_HOST).poll("pid-2", poll_interval=0.01)


@responses.activate
def test_poll_times_out():
    responses.add(responses.GET, f"{_HOST}/history/pid-3", json={}, status=200)
    client = cc.ComfyUIClient(_HOST, timeout=0.05)
    with pytest.raises(cc.ComfyUIError):
        client.poll("pid-3", poll_interval=0.01)


def test_first_output_image():
    result = {"outputs": {"9": {"images": [{"filename": "out.png", "subfolder": "s", "type": "output"}]}}}
    assert cc.first_output_image(result) == ("out.png", "s", "output")


def test_first_output_image_raises_when_empty():
    with pytest.raises(cc.ComfyUIError):
        cc.first_output_image({"outputs": {"9": {"images": []}}})


@responses.activate
def test_download_image_returns_bytes():
    responses.add(responses.GET, f"{_HOST}/view", body=b"\x89PNG-data", status=200)
    data = cc.ComfyUIClient(_HOST).download_image("out.png", subfolder="s")
    assert data == b"\x89PNG-data"


@responses.activate
def test_check_server_true_on_200():
    responses.add(responses.GET, f"{_HOST}/system_stats", json={"ok": 1}, status=200)
    assert cc.ComfyUIClient(_HOST).check_server() is True
