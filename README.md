# hermes-comfyui-local

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) **image-gen backend
provider** for a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server.
It registers a single `comfyui` backend so Hermes's native `image_generate` tool
renders through your own ComfyUI — no cloud image API, no per-image cost.

This is a **backend provider only**: no agent tools, no `pre_llm_call` hook, no
`*_manage` tools, no bundled skill. Image generation stays on the native tool
path, which keeps the model from trying to drive ComfyUI by hand.

## Install

Requires Hermes Agent 0.15.1+ and a reachable ComfyUI server.

```bash
curl -fsSL https://raw.githubusercontent.com/intarweb/hermes-comfyui-local/main/install.sh | bash
hermes gateway restart
```

Manual:

```bash
mkdir -p ~/.hermes/plugins/image_gen
git clone https://github.com/intarweb/hermes-comfyui-local.git \
  ~/.hermes/plugins/image_gen/comfyui-local
hermes plugins enable image_gen/comfyui-local
hermes gateway restart
```

## Configure

In `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - image_gen/comfyui-local

image_gen:
  provider: comfyui          # select this backend
  comfyui:
    host: http://127.0.0.1:8188   # your ComfyUI server, used VERBATIM
    workflow: flux_dev_txt2img    # optional; default is flux_dev_txt2img
    api_key: ""                   # optional
    timeout: 600                  # seconds
```

Every field can be overridden by an environment variable (precedence:
**env > `image_gen.comfyui.*` > default**):

| var | meaning |
|---|---|
| `COMFYUI_HOST` (or `COMFY_HOST`) | ComfyUI server URL |
| `COMFYUI_API_KEY` (or `COMFY_CLOUD_API_KEY`) | optional API key (sent as `X-API-Key`) |
| `COMFYUI_IMAGE_WORKFLOW` | workflow id to use |

> **The host is used exactly as configured.** This plugin never inspects
> `/system_stats` argv to "discover" a launch port — ComfyUI may report an
> internal port (e.g. `18188`) there while only being reachable on the published
> port (`8188`), and trusting it sends requests to a dead port.

## Bundled workflows

Clean, comment-free API-format graphs in [`workflows/`](workflows):

| id | model | notes |
|---|---|---|
| `flux_dev_txt2img` | Flux Dev (`UNETLoader` + `flux1-dev`) | **default** |
| `sdxl_txt2img` | SDXL (`CheckpointLoaderSimple`) | needs an SDXL checkpoint |
| `sd15_txt2img` | SD 1.5 | smallest / fastest |

Drop your own API-format workflows in
`~/.hermes/image_gen/comfyui/workflows/*.json` and select by filename stem. The
plugin discovers the prompt / negative / latent-size / seed nodes automatically
(`CLIPTextEncode`, `EmptyLatentImage`/`EmptySD3LatentImage`, `KSampler`/
`RandomNoise`) and injects prompt, width/height, and seed at generate time.

### Authoring note

ComfyUI's `/prompt` treats every top-level key as a node and rejects any without
a `class_type` (`"node missing class_type property"`). This plugin strips such
keys (e.g. `_comment`) before submitting, so annotated workflows still run — but
the bundled ones ship comment-free regardless.

## How an image reaches you

`generate()` downloads the PNG to Hermes's image cache
(`$HERMES_HOME/cache/images/`) and returns its **absolute path**. Hermes's
delivery layer emits a `MEDIA:<abs-path>` tag for local files, which the webui and
platform adapters render as inline/native media.

## Development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q            # unit tests mock ComfyUI HTTP + the Hermes runtime; no server needed
```

The ComfyUI HTTP client and workflow injection live in
[`comfyui_client.py`](comfyui_client.py) with no dependency on the Hermes
runtime, so they're testable in isolation; the provider wiring lives in
[`__init__.py`](__init__.py).
