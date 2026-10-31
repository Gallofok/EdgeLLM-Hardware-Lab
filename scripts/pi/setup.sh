#!/usr/bin/env bash
# Run explicitly on the authorized Raspberry Pi; never changes boot or network settings.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/.state"
LOG="$ROOT/.state/setup-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
on_error() {
    local exit_code=$?
    echo "Setup failed at line $1 (exit $exit_code). See private log: $LOG" >&2
    exit "$exit_code"
}
trap 'on_error "$LINENO"' ERR
if [[ $# -gt 0 ]]; then
    echo "Usage: bash scripts/pi/setup.sh (BUILD_JOBS=1, SKIP_APT=0)" >&2
    exit 2
fi
if [[ "$(uname -m)" != aarch64 ]] || [[ "$(getconf LONG_BIT)" != 64 ]]; then
    echo "ARM64 userspace is required; see docs/pi3_setup.md for safe SD-card migration." >&2
    exit 2
fi
if [[ ! -r /proc/device-tree/model ]] || ! tr -d '\0' < /proc/device-tree/model | grep -q 'Raspberry Pi 3'; then
    echo "This setup targets Raspberry Pi 3 only; verify your device with diagnose.sh." >&2
    exit 2
fi
JOBS="${BUILD_JOBS:-1}"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'BUILD_JOBS must be a positive integer'; exit 2; }
if [[ "${SKIP_APT:-0}" != 1 ]]; then
    SUDO=()
    if [[ "$EUID" -ne 0 ]]; then SUDO=(sudo); fi
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends git cmake build-essential curl ca-certificates python3 python3-matplotlib
fi
PIN="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["llama_commit"])' "$ROOT/config/versions.json")"
SRC="$ROOT/vendor/llama.cpp"
if [[ ! -e "$SRC" ]]; then
    mkdir -p "$SRC"
    git -C "$SRC" init
    git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
[[ -z "$(git -C "$SRC" status --porcelain)" ]] || { echo 'Refusing to alter a dirty llama.cpp checkout'; exit 1; }
if ! git -C "$SRC" cat-file -e "$PIN^{commit}" 2>/dev/null; then
    git -C "$SRC" fetch --depth 1 origin "$PIN"
fi
git -C "$SRC" checkout --detach "$PIN"
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$PIN" ]]
cmake -S "$SRC" -B "$ROOT/build/llama" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a \
    -DGGML_CPU=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF \
    -DGGML_OPENCL=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build "$ROOT/build/llama" --target llama-bench --parallel "$JOBS"
"$ROOT/build/llama/bin/llama-bench" --help > "$ROOT/.state/llama-bench-help.txt" 2>&1
python3 "$ROOT/scripts/pi/model.py"
python3 "$ROOT/scripts/pi/diagnose.py" --output "$ROOT/.state/environment.json"
python3 - "$ROOT" <<'PY'
import json, pathlib, platform, subprocess, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.pi.common import sha256, utc_now, write_json
versions = json.loads((root / 'config/versions.json').read_text())
write_json(root / '.state/setup.json', {
    'timestamp': utc_now(), 'versions': versions,
    'binary_sha256': sha256(root / 'build/llama/bin/llama-bench'),
    'cmake_cache_sha256': sha256(root / 'build/llama/CMakeCache.txt'),
    'compiler': subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
    'cmake': subprocess.check_output(['cmake', '--version'], text=True).splitlines()[0],
    'python': platform.python_version(),
    'build_flags': {'cpu_arm_arch': 'armv8-a', 'native': False, 'gpu': False},
})
PY
echo 'Setup complete. Run: bash scripts/pi/benchmark.sh --suite baseline'
