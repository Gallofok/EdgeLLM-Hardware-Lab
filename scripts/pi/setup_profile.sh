#!/usr/bin/env bash
# Project-local tool extraction for the observed Debian 13 ARM64 Pi. No sudo/sysctl.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
trap 'echo "Profiling tool setup failed at line $LINENO; inspect package availability and dependencies." >&2' ERR
if [[ $# -gt 0 ]] || [[ "$(uname -m)" != aarch64 ]]; then
    echo 'Usage: bash scripts/pi/setup_profile.sh on the authorized ARM64 Pi' >&2
    exit 2
fi
TOOLS="$ROOT/.state/perf-tools"
mkdir -p "$TOOLS/packages" "$TOOLS/root"
while IFS=$'\t' read -r name version expected; do
    archive=''
    for candidate in "$TOOLS/packages/${name}_"*.deb; do
        if [[ -f "$candidate" ]] && [[ "$(sha256sum "$candidate" | cut -d ' ' -f 1)" == "$expected" ]]; then
            archive="$candidate"
        fi
    done
    if [[ -z "$archive" ]]; then
        (cd "$TOOLS/packages" && apt-get download "$name=$version")
        for candidate in "$TOOLS/packages/${name}_"*.deb; do
            if [[ -f "$candidate" ]] && [[ "$(sha256sum "$candidate" | cut -d ' ' -f 1)" == "$expected" ]]; then
                archive="$candidate"
            fi
        done
    fi
    [[ -n "$archive" ]] || { echo "No verified archive for $name=$version" >&2; exit 1; }
    [[ "$(dpkg-deb -f "$archive" Package)" == "$name" ]]
    [[ "$(dpkg-deb -f "$archive" Version)" == "$version" ]]
    [[ "$(dpkg-deb -f "$archive" Architecture)" == arm64 ]]
    dpkg-deb -x "$archive" "$TOOLS/root"
done < <(python3 - "$ROOT/config/profile_tools.json" <<'PY'
import json, sys
for package in json.load(open(sys.argv[1]))['packages']:
    print(package['name'], package['version'], package['sha256'], sep='\t')
PY
)
python3 - "$ROOT" <<'PY'
import json, os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.pi.common import sha256, utc_now, write_json
tools = root / '.state/perf-tools'
binary = tools / 'root/usr/bin/perf'
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = str(tools / 'root/usr/lib/aarch64-linux-gnu')
env['DEBUGINFOD_URLS'] = ''
version = subprocess.check_output([str(binary), '--version'], env=env, text=True).strip()
write_json(tools / 'setup.json', {
    'timestamp': utc_now(), 'perf_version': version, 'binary_sha256': sha256(binary),
    'tool_config_sha256': sha256(root / 'config/profile_tools.json'),
    'packages': json.loads((root / 'config/profile_tools.json').read_text())['packages'],
    'local_libraries_sha256': {p.name: sha256(p) for p in
        sorted((tools / 'root/usr/lib/aarch64-linux-gnu').glob('*.so.*')) if p.is_file()},
})
print(version + '; verified project-local tools ready. No system packages were installed.')
PY
