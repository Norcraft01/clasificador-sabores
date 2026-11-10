#!/usr/bin/env bash
# Lanzador para Linux / Mac: instala uv si falta y abre la app.
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "Instalando uv (solo la primera vez)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if command -v nvidia-smi >/dev/null 2>&1; then EXTRA=gpu; else EXTRA=cpu; fi
echo "Preparando el entorno (torch $EXTRA). La primera vez tarda varios minutos..."
uv run --extra "$EXTRA" python app.py
