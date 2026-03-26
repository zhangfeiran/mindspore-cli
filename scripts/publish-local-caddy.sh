#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"

if [ -z "${VERSION}" ]; then
  echo "Usage: ./scripts/publish-local-caddy.sh <version>"
  echo "Example: ./scripts/publish-local-caddy.sh v0.4.26"
  exit 1
fi

if [[ "${VERSION}" != v* ]]; then
  echo "Error: version must include leading v, for example v0.4.26" >&2
  exit 1
fi

DIST_DIR="${MSCLI_DIST_DIR:-/home/weizheng/work/ms-cli/dist}"
MIRROR_ROOT="${MSCLI_MIRROR_ROOT:-/opt/downloads/ms-cli/releases}"
TARGET_DIR="${MIRROR_ROOT}/${VERSION}"
LATEST_LINK="${MIRROR_ROOT}/latest"

required_files=(
  "manifest.json"
  "ms-cli-linux-amd64"
  "ms-cli-linux-arm64"
  "ms-cli-darwin-amd64"
  "ms-cli-darwin-arm64"
  "ms-cli-windows-amd64.exe"
)

for file in "${required_files[@]}"; do
  if [ ! -f "${DIST_DIR}/${file}" ]; then
    echo "Error: missing required asset: ${DIST_DIR}/${file}" >&2
    exit 1
  fi
done

echo "Publishing ${VERSION} from ${DIST_DIR} to ${TARGET_DIR}"
sudo mkdir -p "${TARGET_DIR}"
sudo cp "${DIST_DIR}"/* "${TARGET_DIR}/"
sudo chmod -R a+rX "${TARGET_DIR}"
sudo ln -sfn "${TARGET_DIR}" "${LATEST_LINK}"

echo ""
echo "Published ${VERSION} to local Caddy mirror:"
echo "  ${TARGET_DIR}"
echo "Latest link:"
echo "  ${LATEST_LINK} -> ${TARGET_DIR}"
