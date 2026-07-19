"""Test harness: stub the Hermes runtime modules the plugin imports.

The provider imports ``agent.image_gen_provider`` (the ABC + response helpers +
the image cache dir) and ``hermes_cli.config`` (config loader) — both supplied by
the Hermes runtime at deploy, absent in a bare test env. We install lightweight
stand-ins in ``sys.modules`` before the plugin is imported so the orchestration
logic can be exercised without a running Hermes. ``comfyui_client`` has no such
dependency and is imported directly.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

_CACHE_DIR = Path(tempfile.gettempdir()) / "hermes_comfyui_test_cache"


def _install_comfyui_client() -> None:
    """Load comfyui_client.py by path into sys.modules as a top-level module.

    Loading by path (rather than putting the repo root on sys.path) avoids
    pytest trying to import the plugin's root ``__init__.py`` as a bare
    ``__init__`` module.
    """
    if "comfyui_client" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("comfyui_client", _ROOT / "comfyui_client.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_client"] = module
    spec.loader.exec_module(module)


def _install_agent_stub() -> None:
    agent = types.ModuleType("agent")
    igp = types.ModuleType("agent.image_gen_provider")

    igp.DEFAULT_ASPECT_RATIO = "square"

    class ImageGenProvider:  # minimal ABC stand-in
        pass

    def resolve_aspect_ratio(value):
        return value if value in ("landscape", "square", "portrait") else "square"

    def success_response(**kwargs):
        return {"success": True, **kwargs}

    def error_response(**kwargs):
        return {"success": False, **kwargs}

    def _images_cache_dir():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return _CACHE_DIR

    igp.ImageGenProvider = ImageGenProvider
    igp.resolve_aspect_ratio = resolve_aspect_ratio
    igp.success_response = success_response
    igp.error_response = error_response
    igp._images_cache_dir = _images_cache_dir

    agent.image_gen_provider = igp
    sys.modules["agent"] = agent
    sys.modules["agent.image_gen_provider"] = igp


def _install_hermes_cli_stub() -> None:
    hermes_cli = types.ModuleType("hermes_cli")
    config = types.ModuleType("hermes_cli.config")
    config.load_config = lambda: {}  # tests monkeypatch this as needed
    hermes_cli.config = config
    sys.modules["hermes_cli"] = hermes_cli
    sys.modules["hermes_cli.config"] = config


_install_agent_stub()
_install_hermes_cli_stub()
_install_comfyui_client()


def _load_plugin():
    """Import the plugin package (repo-root __init__.py) with relative imports intact."""
    pkgname = "hermes_comfyui_local_pkg"
    if pkgname in sys.modules:
        return sys.modules[pkgname]
    init = _ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        pkgname, init, submodule_search_locations=[str(_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkgname] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin():
    return _load_plugin()


@pytest.fixture
def cache_dir():
    return _CACHE_DIR
