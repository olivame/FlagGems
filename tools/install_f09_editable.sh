#!/usr/bin/env bash
# Portable, backend-agnostic installation script for FlagTree F09 runtime + FlagGems.
# Supports both in-place environment installation and isolated --target directory installation
# without hardcoding vendor names, driver environment variables, or personal paths.
set -euo pipefail

FLAGTREE_DIR=""
FLAGGEMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flagtree) FLAGTREE_DIR="$2"; shift 2 ;;
    --flaggems) FLAGGEMS_DIR="$2"; shift 2 ;;
    --target)   TARGET_DIR="$2"; shift 2 ;;
    --python)   PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --flagtree <FlagTree_repo> [--flaggems <FlagGems_repo>]" \
        "[--target <isolated_dir>] [--python python3]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${FLAGTREE_DIR}" || ! -d "${FLAGTREE_DIR}/python/triton" ]]; then
  echo "ERROR: --flagtree must point to a valid FlagTree repository (containing python/triton)" >&2
  exit 1
fi
if [[ ! -d "${FLAGGEMS_DIR}/src/flag_gems" ]]; then
  echo "ERROR: --flaggems must point to a valid FlagGems repository (containing src/flag_gems)" >&2
  exit 1
fi

F09_TRITON_FILES=(
  knobs.py
  backends/compiler.py
  runtime/__init__.py
  runtime/backend_caps.py
  runtime/_async_compile.py
  runtime/_distributed.py
  runtime/adjust_kernel_param.py
  runtime/autotuner.py
  runtime/bucket_guard.py
  runtime/cache.py
  runtime/errors.py
  runtime/graph_bucketing.py
  runtime/jit.py
  runtime/shape_bucketing.py
  runtime/shape_profiler.py
  flagtune/__init__.py
  flagtune/runtime/autotuner.py
  flagtune/runtime/partition_planner.py
)

INSTALLED_TRITON="$("${PYTHON_BIN}" -c 'import triton, pathlib; print(pathlib.Path(triton.__file__).parent)')"

if [[ -n "${TARGET_DIR}" ]]; then
  mkdir -p "${TARGET_DIR}/triton" "${TARGET_DIR}/flag_gems"
  cp -a "${INSTALLED_TRITON}/." "${TARGET_DIR}/triton/"
  for rel in "${F09_TRITON_FILES[@]}"; do
    mkdir -p "${TARGET_DIR}/triton/$(dirname "${rel}")"
    cp -f "${FLAGTREE_DIR}/python/triton/${rel}" "${TARGET_DIR}/triton/${rel}"
  done
  cp -a "${FLAGGEMS_DIR}/src/flag_gems/." "${TARGET_DIR}/flag_gems/"
  echo "Isolated F09 installation ready at: ${TARGET_DIR}"
  echo "Use: PYTHONPATH=\"${TARGET_DIR}\" ${PYTHON_BIN} ..."
else
  "${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps -e "${FLAGGEMS_DIR}" --quiet
  for rel in "${F09_TRITON_FILES[@]}"; do
    mkdir -p "${INSTALLED_TRITON}/$(dirname "${rel}")"
    cp -f "${FLAGTREE_DIR}/python/triton/${rel}" "${INSTALLED_TRITON}/${rel}"
  done
  echo "Editable F09 installation complete: FlagTree -> ${INSTALLED_TRITON}, FlagGems -> ${FLAGGEMS_DIR}"
fi
