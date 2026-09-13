#!/usr/bin/env bash
# 在一台带 CUDA 的 Linux 机器上从零把环境建好。
#
# 用法：
#   bash setup.sh /opt/hrm                      # 第一个参数是安装根目录，默认 ./hrm
#   TORCH_INDEX=default bash setup.sh /opt/hrm   # 没有 CUDA 的机器
#   PYBIN=/usr/bin/python3.10 bash setup.sh ...  # python3.10 不在 PATH 时
#
# 装完之后：
#   export HRM_LEARNING=<根目录>/hrm-learning
#   export HRM_PYTHON=<根目录>/venv/bin/python
#   $HRM_PYTHON runner/verify.py
set -euo pipefail

ROOT="${1:-$PWD/hrm}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT"
cd "$ROOT"

echo "==> 1/6 克隆三个仓库到锁定的提交"
clone() {  # clone <repo> <commit>
  if [ ! -d "$1" ]; then
    git clone "https://github.com/ertsiger/$1.git" "$1"
  fi
  git -C "$1" fetch --depth 50 origin || true
  git -C "$1" checkout -q "$2"
  echo "    $1 @ $(git -C "$1" rev-parse --short HEAD)"
}
clone hrm-formalism-envs 929605f
clone hrm-learning       23d0e25
clone hrm-minigrid       d0e6763

echo "==> 2/6 打补丁"
git -C hrm-formalism-envs apply --check "$HERE/patches/01-hrm-formalism-envs.patch" \
  && git -C hrm-formalism-envs apply "$HERE/patches/01-hrm-formalism-envs.patch" \
  && echo "    01 已应用" || echo "    01 已是最新，跳过"
git -C hrm-learning apply --check "$HERE/patches/02-hrm-learning.patch" \
  && git -C hrm-learning apply "$HERE/patches/02-hrm-learning.patch" \
  && echo "    02 已应用" || echo "    02 已是最新，跳过"

echo "==> 3/6 放入新增的算法文件"
cp "$HERE"/src/*.py hrm-learning/src/reinforcement_learning/
echo "    已复制 $(ls "$HERE"/src/*.py | wc -l | tr -d ' ') 个文件"

echo "==> 4/6 建虚拟环境（Python 3.10）"
PYBIN="${PYBIN:-python3.10}"
command -v "$PYBIN" >/dev/null || { echo "找不到 $PYBIN，请先装 Python 3.10 或设 PYBIN"; exit 1; }
"$PYBIN" -m venv venv
venv/bin/python -m pip install -q --upgrade pip setuptools wheel

echo "==> 5/6 装依赖（顺序重要，见下）"
# gym 0.15.3 必须先装，它会从源码构建
venv/bin/python -m pip install -q "gym==0.15.3"
venv/bin/python -m pip install -q -e hrm-minigrid
venv/bin/python -m pip install -q --no-deps -e hrm-formalism-envs
# CUDA 版 torch。TORCH_INDEX 换成与你驱动匹配的版本；
# 设成 default 则用 PyPI 默认源（没有 CUDA 的机器上这样装）。
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
if [ "$TORCH_INDEX" = "default" ]; then
  venv/bin/python -m pip install -q torch
else
  venv/bin/python -m pip install -q torch --index-url "$TORCH_INDEX"
fi
venv/bin/python -m pip install -q matplotlib pygame pandas tqdm xlsxwriter requests
# numpy 必须放最后，而且必须 < 1.24：源码有 5 处用 np.int / np.bool，
# numpy 1.24 移除了这些别名；装 torch 会把 numpy 顶到 2.x。
venv/bin/python -m pip install -q "numpy==1.23.5"

echo "==> 6/6 自检"
venv/bin/python - <<'PY'
import torch, numpy, gym
print(f"    torch {torch.__version__}  CUDA 可用 {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
print(f"    numpy {numpy.__version__}  (必须 < 1.24)")
print(f"    gym {gym.__version__}  (必须是 0.15.3)")
import gym_minigrid, gym_hierarchical_subgoal_automata
print("    两个环境包导入正常")
PY

cat <<EOF

安装完成。接下来：

    export HRM_LEARNING=$ROOT/hrm-learning
    export HRM_PYTHON=$ROOT/venv/bin/python
    \$HRM_PYTHON "$HERE/tools/verify.py"

verify.py 全部通过之后再跑网格，别跳过。
EOF
