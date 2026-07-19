"""Minimal ComfyUI REST client + API-format workflow injection.

Clean-room implementation of the ComfyUI HTTP dance (``POST /prompt`` ->
poll ``/history/{id}`` -> ``GET /view``) and the field-injection map for
API-format workflow graphs. No process autostart, no port discovery: the
configured host is used **verbatim** (see ``__init__._comfy_host``). This module
has no dependency on the Hermes runtime, so it is unit-testable on its own.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://127.0.0.1:8188"
DEFAULT_TIMEOUT = 600.0


class ComfyUIError(RuntimeError):
    """Any failure talking to ComfyUI or preparing a workflow for it."""


def _random_seed() -> int:
    return random.randint(0, 2**32 - 1)


def _is_link(value: Any) -> bool:
    """True if a node input is wired to another node: ``[<from_node_id>, <slot>]``."""
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


def strip_non_nodes(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``workflow`` containing only real nodes.

    ComfyUI's ``/prompt`` endpoint treats **every** top-level key as a node and
    rejects any that lacks a ``class_type`` (``"<id>: node missing class_type
    property"``). Workflow authors routinely add annotation keys such as
    ``_comment``; this drops any top-level entry whose value is not a dict
    containing ``class_type``. This is the single most common reason a
    hand-authored workflow fails to submit.
    """
    return {
        node_id: node
        for node_id, node in workflow.items()
        if isinstance(node, dict) and "class_type" in node
    }


def iter_nodes(workflow: Dict[str, Any]):
    """Yield ``(node_id, node)`` for every real node (skips ``_`` annotations)."""
    for node_id, node in workflow.items():
        if node_id.startswith("_") or not isinstance(node, dict):
            continue
        if "class_type" in node:
            yield node_id, node


def infer_workflow_spec(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort injection map for an API-format ComfyUI graph.

    Locates the positive/negative prompt text inputs, the latent size node, and
    every seed field so :func:`inject_workflow` knows where to write.
    """
    prompt_node: Optional[Tuple[str, str]] = None
    negative_node: Optional[Tuple[str, str]] = None
    latent_node: Optional[Tuple[str, str, str]] = None
    seed_fields: List[Tuple[str, str]] = []
    base_size = 1024

    for node_id, node in iter_nodes(workflow):
        cls = node.get("class_type")
        inputs = node.get("inputs") or {}
        title = str((node.get("_meta") or {}).get("title") or "").lower()

        if cls == "CLIPTextEncode":
            if "negative" in title and negative_node is None:
                negative_node = (node_id, "text")
            elif prompt_node is None and "negative" not in title:
                prompt_node = (node_id, "text")

        if cls in ("EmptyLatentImage", "EmptySD3LatentImage") and latent_node is None:
            latent_node = (node_id, "width", "height")
            try:
                base_size = int(inputs.get("width") or inputs.get("height") or base_size)
            except (TypeError, ValueError):
                pass

        if cls in ("KSampler", "KSamplerAdvanced") and "seed" in inputs:
            seed_fields.append((node_id, "seed"))
        if cls == "RandomNoise" and "noise_seed" in inputs:
            seed_fields.append((node_id, "noise_seed"))

    if prompt_node is None:
        for node_id, node in iter_nodes(workflow):
            if node.get("class_type") == "CLIPTextEncode":
                prompt_node = (node_id, "text")
                break

    return {
        "prompt": prompt_node,
        "negative": negative_node,
        "latent": latent_node,
        "seed_fields": seed_fields,
        "base_size": base_size,
    }


def inject_workflow(
    workflow: Dict[str, Any],
    *,
    prompt: str,
    width: int,
    height: int,
    seed: Optional[int] = None,
    spec: Dict[str, Any],
    negative_prompt: str = "",
) -> Dict[str, Any]:
    """Return a deep copy of ``workflow`` with prompt/size/seed injected per ``spec``."""
    wf = copy.deepcopy(workflow)

    prompt_target = spec.get("prompt")
    if not prompt_target:
        raise ComfyUIError("Workflow has no injectable prompt node (CLIPTextEncode)")
    prompt_node, prompt_field = prompt_target
    wf[prompt_node]["inputs"][prompt_field] = prompt

    negative_target = spec.get("negative")
    if negative_target and negative_prompt:
        neg_node, neg_field = negative_target
        wf[neg_node]["inputs"][neg_field] = negative_prompt

    latent_node = spec.get("latent")
    if latent_node:
        node_id, w_field, h_field = latent_node
        wf[node_id]["inputs"][w_field] = width
        wf[node_id]["inputs"][h_field] = height

    if seed is None:
        seed = _random_seed()
    for node_id, field in list(spec.get("seed_fields") or []):
        if node_id in wf and isinstance(wf[node_id], dict):
            inputs = wf[node_id].setdefault("inputs", {})
            # Never clobber a wired input (e.g. seed fed from another node).
            if field in inputs and not _is_link(inputs.get(field)):
                inputs[field] = seed
    return wf


def first_output_image(result: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return ``(filename, subfolder, type)`` of the first output image in a run."""
    outputs = result.get("outputs") if isinstance(result, dict) else None
    if not isinstance(outputs, dict):
        raise ComfyUIError("ComfyUI result missing outputs")
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        images = node_output.get("images")
        if not isinstance(images, list) or not images:
            continue
        first = images[0]
        if not isinstance(first, dict):
            continue
        filename = str(first.get("filename") or "").strip()
        if filename:
            return (
                filename,
                str(first.get("subfolder") or ""),
                str(first.get("type") or "output"),
            )
    raise ComfyUIError("ComfyUI outputs contained no images")


def aspect_to_size(aspect_ratio: str, base: int = 1024) -> Tuple[int, int]:
    """Map a Hermes aspect string (landscape/square/portrait) to width/height."""
    if aspect_ratio == "portrait":
        w, h = int(base * 0.75), base
    elif aspect_ratio == "landscape":
        w, h = base, int(base * 0.75)
    else:  # square (also the default fallback)
        w, h = base, base
    w = max(256, (w // 8) * 8)
    h = max(256, (h // 8) * 8)
    return w, h


class ComfyUIClient:
    """Thin synchronous client for a self-hosted ComfyUI server.

    The ``host`` is used exactly as given — this client never inspects
    ``/system_stats`` argv to "discover" a launch port, which can report an
    internal port (e.g. 18188) that isn't the reachable published one (8188).
    """

    def __init__(self, host: str, api_key: str = "", timeout: float = DEFAULT_TIMEOUT):
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def check_server(self) -> bool:
        """True if the server answers ``GET /system_stats`` with 200."""
        try:
            resp = requests.get(f"{self.host}/system_stats", headers=self._headers(), timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def submit(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow and return its ``prompt_id``.

        Non-node top-level keys are stripped first (see :func:`strip_non_nodes`).
        """
        body = {"prompt": strip_non_nodes(workflow), "client_id": self.client_id}
        resp = requests.post(f"{self.host}/prompt", headers=self._headers(), json=body, timeout=60)
        if resp.status_code >= 400:
            raise ComfyUIError(
                f"ComfyUI prompt submit failed ({resp.status_code}): {resp.text[:500]}"
            )
        data = resp.json()
        if not isinstance(data, dict):
            raise ComfyUIError("ComfyUI prompt submit returned non-object JSON")
        node_errors = data.get("node_errors") or {}
        if node_errors:
            raise ComfyUIError(
                f"ComfyUI workflow validation failed: {json.dumps(node_errors)[:500]}"
            )
        prompt_id = data.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyUIError("ComfyUI prompt submit missing prompt_id")
        return prompt_id

    def poll(self, prompt_id: str, *, poll_interval: float = 1.0) -> Dict[str, Any]:
        """Poll ``/history/{prompt_id}`` until the run completes; return its entry."""
        deadline = time.monotonic() + self.timeout
        interval = poll_interval
        while time.monotonic() < deadline:
            resp = requests.get(
                f"{self.host}/history/{prompt_id}", headers=self._headers(), timeout=30
            )
            if resp.status_code == 200:
                data = resp.json() if resp.content else {}
                entry = data.get(prompt_id) if isinstance(data, dict) else None
                if isinstance(entry, dict):
                    status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
                    if status.get("status_str") == "error":
                        raise ComfyUIError(f"ComfyUI execution error: {json.dumps(status)[:500]}")
                    if status.get("completed"):
                        return entry
            time.sleep(interval)
            interval = min(5.0, interval * 1.3)
        raise ComfyUIError(f"ComfyUI job {prompt_id} timed out after {int(self.timeout)}s")

    def download_image(
        self, filename: str, subfolder: str = "", file_type: str = "output"
    ) -> bytes:
        """Fetch raw image bytes from ``GET /view``."""
        params = urlencode({"filename": filename, "subfolder": subfolder, "type": file_type})
        resp = requests.get(
            f"{self.host}/view?{params}",
            headers=self._headers(),
            timeout=120,
            allow_redirects=True,
        )
        if resp.status_code >= 400:
            raise ComfyUIError(f"ComfyUI download failed ({resp.status_code}) for {filename}")
        return resp.content


def discover_workflows(plugin_workflow_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return ``{workflow_id: metadata}`` from the bundled dir + the user dir.

    User workflows live in ``~/.hermes/image_gen/comfyui/workflows``; a user file
    whose stem collides with a bundled one is exposed as ``user-<stem>``.
    """
    catalog: Dict[str, Dict[str, Any]] = {}

    def _scan(directory: Path, *, source: str) -> None:
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.json")):
            workflow_id = path.stem
            if workflow_id in catalog:
                workflow_id = f"{source}-{path.stem}"
            try:
                with path.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    continue
            except Exception:
                continue
            spec = infer_workflow_spec(data)
            catalog[workflow_id] = {
                "display": path.stem.replace("_", " ").title(),
                "workflow": path.name,
                "path": str(path),
                "source": source,
                "speed": "local",
                "strengths": f"ComfyUI workflow ({source})",
                **spec,
            }

    _scan(plugin_workflow_dir, source="bundled")
    _scan(user_workflow_dir(), source="user")
    return catalog


def user_workflow_dir() -> Path:
    return Path.home() / ".hermes" / "image_gen" / "comfyui" / "workflows"


def read_workflow(path: str) -> Dict[str, Any]:
    """Load a workflow JSON file into a dict."""
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ComfyUIError(f"workflow JSON must be an object: {path}")
    return data
