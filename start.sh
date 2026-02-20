#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer is intended for Ubuntu/Linux." >&2
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found. This installer requires an Ubuntu-based system." >&2
  exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "This script needs sudo privileges to install required packages." >&2
    exit 1
  fi
fi

if command -v uv >/dev/null 2>&1; then
  echo "uv is already installed at $(command -v uv)"
  exit 0
fi

echo "Updating package index..."
${SUDO} apt-get update -qq

echo "Installing curl and CA certificates..."
${SUDO} apt-get install -y ca-certificates curl >/dev/null

echo "Running the official uv installer..."
curl -LsSf https://astral.sh/uv/install.sh | sh

if command -v uv >/dev/null 2>&1; then
  echo "uv installed successfully at $(command -v uv)"
else
  echo "uv installation finished. Ensure ${HOME}/.local/bin is in your PATH." >&2
fi
