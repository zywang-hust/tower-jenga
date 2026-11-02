#!/usr/bin/env bash
# One environment for training, serving, offline evaluation and the CPU tests: OpenPI at a pinned commit with
# patches/openpi-tower.patch applied, its exact lockfile, this package on top, and the official starter's wire codec
# (external/tower-starter, imported by jenga.server). Needs git, uv and network access to GitHub, the package index
# (UV_DEFAULT_INDEX, default PyPI) and challenge.tower-benchmark.com (or TOWER_STARTER_ZIP=<local copy of the zip>).
# The local simulator (sim/) uses two more environments; see sim/isaac.sh and the README.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
project_root="$PWD"
upstream_revision="215abfb217dbac7d5f1273282331b9b1866c0479"
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p external
if [[ ! -d external/openpi/.git ]]; then
    git init -q external/openpi
    git -C external/openpi remote add origin https://github.com/Physical-Intelligence/openpi.git
    git -C external/openpi fetch --depth 1 origin "$upstream_revision"
    git -C external/openpi checkout --detach "$upstream_revision"
fi
if [[ "$(git -C external/openpi rev-parse HEAD)" != "$upstream_revision" ]]; then
    echo "external/openpi has a different revision; preserve it and reconcile explicitly." >&2
    exit 1
fi
if git -C external/openpi apply --check "$project_root/patches/openpi-tower.patch" 2>/dev/null; then
    git -C external/openpi apply "$project_root/patches/openpi-tower.patch"
elif ! git -C external/openpi apply --reverse --check "$project_root/patches/openpi-tower.patch" 2>/dev/null; then
    echo "OpenPI patch conflicts with local changes; no files overwritten." >&2
    exit 1
fi
uv python install 3.11
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.11 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use a Python 3.11 venv"'
mkdir -p runs
# Exact upstream versions, fetched from the configured index rather than the registry URLs in the upstream lockfile.
uv export --project external/openpi --frozen --no-dev --no-emit-workspace --no-hashes \
    --format requirements-txt --output-file runs/openpi-requirements.txt >/dev/null
package_index="${UV_DEFAULT_INDEX:-https://pypi.org/simple}"
uv pip sync --python .venv/bin/python --index-url "$package_index" runs/openpi-requirements.txt
uv pip install --python .venv/bin/python --index-url "$package_index" --no-deps \
    -e external/openpi -e external/openpi/packages/openpi-client
uv pip install --python .venv/bin/python --index-url "$package_index" -e '.[dev]'
uv pip check --python .venv/bin/python
if [[ ! -f external/tower-starter/tower_protocol.py ]]; then
    .venv/bin/python scripts/fetch_starter.py ${TOWER_STARTER_ZIP:+--archive "$TOWER_STARTER_ZIP"}
fi
echo "Ready: source $project_root/.venv/bin/activate"

