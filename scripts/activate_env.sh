# Source this file from bash or zsh: source scripts/activate_env.sh
if [ -n "${BASH_VERSION:-}" ]; then
    _starvla_source="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _starvla_source="${(%):-%x}"
else
    printf 'Please source this file from bash or zsh.\n' >&2
    return 1
fi
_starvla_root="$(cd -- "$(dirname -- "${_starvla_source}")/.." && pwd)"
source "${_starvla_root}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${_starvla_root}/.cache/huggingface"
export TORCH_HOME="${_starvla_root}/.cache/torch"
export TRITON_CACHE_DIR="${_starvla_root}/.cache/triton"
export MPLCONFIGDIR="${_starvla_root}/.cache/matplotlib"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_CACHE_DIR="${_starvla_root}/.cache/pip"
export UV_CACHE_DIR="${_starvla_root}/.cache/uv"
export TORCH_EXTENSIONS_DIR="${_starvla_root}/.cache/torch_extensions"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,.tsinghua.edu.cn,.huaweicloud.com"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,.tsinghua.edu.cn,.huaweicloud.com"
unset _starvla_root _starvla_source
