"""hermes-comfyui-local — a Hermes ``kind: backend`` image-gen provider for local ComfyUI.

Registers a single ComfyUI backend so Hermes's native ``image_generate`` tool
renders through a local ComfyUI server. This is a **backend provider only** — it
ships no agent tools, no ``pre_llm_call`` hook, no ``*_manage`` tools, and no
bundled skill. Those extra surfaces let the model drive ComfyUI by hand and go
off-rails; a backend provider keeps generation on the native tool path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

# Hermes loads this plugin dir as a package, so `.comfyui_client` is the correct
# import. Fall back to a top-level import for tooling that imports this file
# outside a package context (e.g. a test collector).
try:
    from .comfyui_client import (
        ComfyUIClient,
        ComfyUIError,
        aspect_to_size,
        discover_workflows,
        first_output_image,
        infer_workflow_spec,
        inject_workflow,
        read_workflow,
    )
except ImportError:  # pragma: no cover - exercised only outside a package context
    from comfyui_client import (
        ComfyUIClient,
        ComfyUIError,
        aspect_to_size,
        discover_workflows,
        first_output_image,
        infer_workflow_spec,
        inject_workflow,
        read_workflow,
    )

logger = logging.getLogger(__name__)

# Flux is the default: it needs only the Flux stack (UNETLoader + flux1-dev,
# the two CLIP encoders, and a VAE) — no checkpoint file — so it runs on servers
# that have no SDXL/SD1.5 checkpoint installed. SDXL and SD 1.5 ship for hosts
# that do have those checkpoints.
DEFAULT_MODEL = "flux_dev_txt2img"

_PLUGIN_DIR = Path(__file__).resolve().parent
_WORKFLOW_DIR = _PLUGIN_DIR / "workflows"


# --------------------------------------------------------------------------- #
# Config resolution. Precedence: env > image_gen.comfyui.* (config.yaml) >
# default. The host is used verbatim — never probed/rewritten.
# --------------------------------------------------------------------------- #
def _load_comfy_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if not isinstance(section, dict):
            return {}
        nested = section.get("comfyui")
        return nested if isinstance(nested, dict) else {}
    except Exception as exc:  # not inside a Hermes runtime, or no config yet
        logger.debug("Could not load image_gen.comfyui config: %s", exc)
        return {}


def _comfy_host() -> str:
    env = (os.environ.get("COMFYUI_HOST") or os.environ.get("COMFY_HOST") or "").strip()
    if env:
        return env.rstrip("/")
    host = str(_load_comfy_config().get("host") or "").strip()
    return host.rstrip("/") if host else "http://127.0.0.1:8188"


def _comfy_api_key() -> str:
    env = (os.environ.get("COMFYUI_API_KEY") or os.environ.get("COMFY_CLOUD_API_KEY") or "").strip()
    if env:
        return env
    return str(_load_comfy_config().get("api_key") or "")


def _comfy_timeout() -> float:
    try:
        return float(_load_comfy_config().get("timeout", 600.0))
    except (TypeError, ValueError):
        return 600.0


def _workflow_catalog() -> Dict[str, Dict[str, Any]]:
    return discover_workflows(_WORKFLOW_DIR)


def _resolve_model(requested: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Pick a workflow id + its spec. Precedence: requested > env > config > default."""
    catalog = _workflow_catalog()
    if not catalog:
        raise KeyError("no ComfyUI workflows found")

    candidates: List[str] = []
    if requested:
        candidates.append(requested.strip())
    env_override = (os.environ.get("COMFYUI_IMAGE_WORKFLOW") or "").strip()
    if env_override:
        candidates.append(env_override)
    cfg = _load_comfy_config()
    for key in ("model", "workflow"):
        val = cfg.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())

    for candidate in candidates:
        if candidate in catalog:
            return candidate, catalog[candidate]
        stem = Path(candidate).stem
        if stem in catalog:
            return stem, catalog[stem]

    if DEFAULT_MODEL in catalog:
        return DEFAULT_MODEL, catalog[DEFAULT_MODEL]
    first = next(iter(catalog))
    return first, catalog[first]


class ComfyUIImageGenProvider(ImageGenProvider):
    """Local ComfyUI text-to-image backend."""

    @property
    def name(self) -> str:
        return "comfyui"

    @property
    def display_name(self) -> str:
        return "ComfyUI (local)"

    def is_available(self) -> bool:
        return ComfyUIClient(_comfy_host(), _comfy_api_key(), timeout=5).check_server()

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta.get("display", model_id),
                "speed": meta.get("speed", "local"),
                "strengths": meta.get("strengths", "ComfyUI workflow"),
                "price": "local",
            }
            for model_id, meta in _workflow_catalog().items()
        ]

    def default_model(self) -> Optional[str]:
        try:
            model_id, _ = _resolve_model()
            return model_id
        except KeyError:
            return None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "ComfyUI (local)",
            "badge": "free · local",
            "tag": "Text-to-image via a local ComfyUI server (Flux Dev / SDXL / SD 1.5)",
            "env_vars": [
                {
                    "key": "COMFYUI_HOST",
                    "prompt": "ComfyUI server URL (default http://127.0.0.1:8188)",
                    "url": "https://github.com/comfyanonymous/ComfyUI",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        aspect = resolve_aspect_ratio(aspect_ratio)

        try:
            requested = str(kwargs.get("model")).strip() if kwargs.get("model") else None
            model_id, spec = _resolve_model(requested)
        except KeyError as exc:
            return error_response(
                error=str(exc),
                error_type="config_error",
                provider=self.name,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not prompt or not str(prompt).strip():
            return error_response(
                error="Prompt is required",
                error_type="validation_error",
                provider=self.name,
                model=model_id,
                prompt=prompt or "",
                aspect_ratio=aspect,
            )

        host = _comfy_host()
        client = ComfyUIClient(host, _comfy_api_key(), timeout=_comfy_timeout())
        if not client.check_server():
            return error_response(
                error=f"ComfyUI server not reachable at {host}",
                error_type="backend_unavailable",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            workflow = read_workflow(str(spec.get("path")))
        except Exception as exc:
            return error_response(
                error=f"Failed to load ComfyUI workflow: {exc}",
                error_type="config_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Re-infer if the cached spec lacks a prompt target (defensive).
        if not spec.get("prompt"):
            spec = infer_workflow_spec(workflow)

        width, height = aspect_to_size(aspect, int(spec.get("base_size", 1024)))
        try:
            prepared = inject_workflow(
                workflow,
                prompt=str(prompt).strip(),
                width=width,
                height=height,
                seed=kwargs.get("seed"),
                spec=spec,
                negative_prompt=str(kwargs.get("negative_prompt") or ""),
            )
        except Exception as exc:
            return error_response(
                error=f"Failed to prepare ComfyUI workflow: {exc}",
                error_type="validation_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            prompt_id = client.submit(prepared)
            result = client.poll(prompt_id)
            filename, subfolder, file_type = first_output_image(result)
            image_bytes = client.download_image(filename, subfolder=subfolder, file_type=file_type)
        except ComfyUIError as exc:
            return error_response(
                error=str(exc),
                error_type="api_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            return error_response(
                error=f"ComfyUI generation failed: {exc}",
                error_type="provider_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Save into Hermes's image cache and return the ABSOLUTE PATH. Hermes's
        # delivery layer emits a `MEDIA:<abs-path>` tag for local files, which
        # the webui renders inline; a markdown `![](path)` would show as raw text.
        from agent.image_gen_provider import _images_cache_dir

        out_dir = _images_cache_dir()
        out_path = out_dir / f"comfyui_{model_id}_{prompt_id[:8]}.png"
        out_path.write_bytes(image_bytes)

        return success_response(
            image=str(out_path),
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra={
                "workflow": spec.get("workflow") or model_id,
                "comfy_prompt_id": prompt_id,
                "width": width,
                "height": height,
                "host": host,
            },
        )


def register(ctx) -> None:
    """Plugin entry point — called once at load time."""
    ctx.register_image_gen_provider(ComfyUIImageGenProvider())
