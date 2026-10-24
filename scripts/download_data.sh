#!/usr/bin/env bash
# Script para baixar e descompactar os dados da competição Kaggle Playground Series S6E10
set -euo pipefail

COMPETITION="playground-series-s6e10"
TARGET_DIR="data/raw"

mkdir -p "$TARGET_DIR"

echo "=== Verificando instalação da CLI do Kaggle ==="
if ! command -v kaggle &> /dev/null; then
    echo "Erro: Kaggle CLI não encontrada no PATH."
    echo "Instale com: pip install kaggle (ou uv pip install kaggle)"
    echo "E certifique-se de configurar seu token em ~/.kaggle/kaggle.json"
    exit 1
fi

echo "=== Baixando dados da competição: $COMPETITION ==="
kaggle competitions download -c "$COMPETITION" -p "$TARGET_DIR"

ZIP_FILE="$TARGET_DIR/${COMPETITION}.zip"
if [ -f "$ZIP_FILE" ]; then
    echo "=== Descompactando $ZIP_FILE em $TARGET_DIR ==="
    unzip -o "$ZIP_FILE" -d "$TARGET_DIR"
    rm "$ZIP_FILE"
    echo "=== Download e descompactação concluídos com sucesso! ==="
    ls -lh "$TARGET_DIR"
else
    echo "Aviso: Arquivo zip não encontrado no diretório $TARGET_DIR."
fi
