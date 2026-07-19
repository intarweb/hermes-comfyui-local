#!/usr/bin/env bash
# hermes-comfyui-local installer.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/intarweb/hermes-comfyui-local/main/install.sh | bash
#
# Requires Hermes Agent 0.15.1+, git, and the hermes CLI. Installs the plugin
# under ~/.hermes/plugins/image_gen/ (Hermes expects image-gen backends there,
# not in the flat plugins folder).
set -euo pipefail

GITHUB_REPO="intarweb/hermes-comfyui-local"
NESTED_FOLDER="comfyui-local"
PLUGIN_ID="image_gen/${NESTED_FOLDER}"

HERMES_HOME="${HERMES_HOME:-${HOME}/.hermes}"
PLUGINS_DIR="${HERMES_HOME}/plugins"

die() { echo "install.sh: $*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is not installed."
command -v hermes >/dev/null 2>&1 || die "hermes CLI not found. Install Hermes Agent first: curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash"

dest="${PLUGINS_DIR}/image_gen/${NESTED_FOLDER}"
mkdir -p "${PLUGINS_DIR}/image_gen"

if [[ -d "${dest}/.git" ]]; then
  echo "==> Updating ${PLUGIN_ID}"
  git -C "${dest}" pull --ff-only
else
  echo "==> Cloning ${PLUGIN_ID} (${GITHUB_REPO})"
  git clone --depth 1 "https://github.com/${GITHUB_REPO}.git" "${dest}"
fi

hermes plugins enable "${PLUGIN_ID}"

echo ""
echo "Installed and enabled: ${PLUGIN_ID}"
echo "Set image_gen.provider: comfyui (and image_gen.comfyui.host) in ${HERMES_HOME}/config.yaml,"
echo "then restart the gateway: hermes gateway restart"
