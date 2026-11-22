#!/usr/bin/env bash
# Portable, backend-agnostic installation script for FlagTree F09 runtime + FlagGems.
# Recommends isolated --target directory installation so existing site-packages triton
# wheels remain untouched. In-place site-packages overlay requires explicit opt-in via
# --allow-in-place-triton-overlay.
set -euo pipefail

FLAGTREE_DIR=""
FLAGGEMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
ALLOW_IN_PLACE_TRITON_OVERLAY=0

usage() {
  cat <<'USAGE'
Usage: install_f09_editable.sh --flagtree <FlagTree_repo> \
  [--flaggems <FlagGems_repo>] [--target <isolated_dir>] \
  [--python python3] [--allow-in-place-triton-overlay]

Modes:
  --target <dir>                     (Recommended) Copy prebuilt triton + FlagTree F09
                                     Python runtime + FlagGems into an isolated directory
                                     without modifying system site-packages.
  --allow-in-place-triton-overlay    Explicit opt-in required to overwrite F09 Python
                                     files inside installed site-packages/triton/ when
                                     --target is omitted.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flagtree) FLAGTREE_DIR="$2"; shift 2 ;;
    --flaggems) FLAGGEMS_DIR="$2"; shift 2 ;;
    --target)   TARGET_DIR="$2"; shift 2 ;;
    --python)   PYTHON_BIN="$2"; shift 2 ;;
    --allow-in-place-triton-overlay) ALLOW_IN_PLACE_TRITON_OVERLAY=1; shift 1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
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
  flagtune/runtime/comm_profiler.py
  flagtune/runtime/parallel_executor.py
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
  if [[ "${ALLOW_IN_PLACE_TRITON_OVERLAY}" -ne 1 ]]; then
    echo "REFUSED: In-place installation would overwrite F09 Python files inside" >&2
    echo "  ${INSTALLED_TRITON}" >&2
    echo "which mutates the installed triton wheel in site-packages." >&2
    echo "" >&2
    echo "Recommended (non-destructive isolated install):" >&2
    echo "  bash \"$0\" --flagtree \"${FLAGTREE_DIR}\" --flaggems \"${FLAGGEMS_DIR}\" --target <isolated_dir>" >&2
    echo "" >&2
    echo "If in-place overlay is explicitly intended (e.g. disposable container)," >&2
    echo "re-run with --allow-in-place-triton-overlay." >&2
    exit 2
  fi
  echo "WARNING: Proceeding with explicit --allow-in-place-triton-overlay into ${INSTALLED_TRITON}" >&2
  "${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps -e "${FLAGGEMS_DIR}" --quiet
  for rel in "${F09_TRITON_FILES[@]}"; do
    mkdir -p "${INSTALLED_TRITON}/$(dirname "${rel}")"
    cp -f "${FLAGTREE_DIR}/python/triton/${rel}" "${INSTALLED_TRITON}/${rel}"
  done
  echo "Editable F09 installation complete: FlagTree -> ${INSTALLED_TRITON}, FlagGems -> ${FLAGGEMS_DIR}"
fi
