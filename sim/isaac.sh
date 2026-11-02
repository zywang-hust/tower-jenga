#!/usr/bin/env bash
# Run a script with the Isaac Sim 5.1 venv (ISAAC_PYTHON: its python) and the EULA accepted. ISAAC_SYSLIBS
# (optional): root of X/GL libraries a headless image may lack (libXt, libGLU, libOpenGL, e.g. unpacked Ubuntu noble
# debs), prepended to LD_LIBRARY_PATH. Kit's extension registry needs network access (HTTP(S)_PROXY is inherited).
#   sim/isaac.sh sim/replay.py <args>         all GPUs visible; pin with active_gpu/physics_gpu in the app config
#   GPU=2 sim/isaac.sh ...                    also restrict CUDA (Vulkan still sees every GPU)
set -euo pipefail
if [[ -n ${JENGA_ENV:-} ]]; then source "$JENGA_ENV"; fi  # optional site setup (toolchains, caches, proxies)
export OMNI_KIT_ACCEPT_EULA=YES
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
if [ -n "${ISAAC_SYSLIBS:-}" ]; then
  export LD_LIBRARY_PATH="$ISAAC_SYSLIBS/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "${ISAAC_PYTHON:?set ISAAC_PYTHON to the python of the Isaac Sim 5.1 venv}" "$@"
