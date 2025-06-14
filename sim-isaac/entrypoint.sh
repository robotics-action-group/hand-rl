#!/bin/bash

set -eo pipefail
if [ "$1" = "job" ]; then
    for f in /docker-entrypoint-init.d/*; do
        case "$f" in
            *.sh)     echo "$0: running $f"; . "$f" ;;
            *)        echo "$0: ignoring $f"        ;;
        esac
        
        if [ -d "$f" ]; then
            cp -rf "$f" /workspace/
        fi
    done
fi

echo "Installing dependencies..."

shopt -s expand_aliases
cd /workspace/c3po/source/c3po
source ${HOME}/.bashrc && \
# pip install -e .

cd /workspace/c3po/source/c3po_utils
pip install -e . --quiet

cd /workspace
# git clone https://github.com/svaichu/rldata.git
cd rldata
pip install -e . --quiet

# cd /workspace
# pip uninstall -y rsl-rl-lib
# git clone https://github.com/svaichu/rsl_rl.git
# cd rsl_rl
# git checkout feat/add-video-wandb
# pip install -e . --quiet
# cd /workspace
# git clone https://github.com/svaichu/lerobot.git
# cd lerobot
# git checkout fix/torch-version
# pip install -e .
# cd /workspace
# git clone https://github.com/NVIDIA/Isaac-GR00T.git
# cd Isaac-GR00T
# pip install -e .
# pip install tensordict torchrl

echo "All dependencies installed successfully."

# if deploy arg is passed, run this file
if [ "$1" = "job" ]; then
    echo "Running job mode"
    cd /workspace/c3po 
    FOLDER_NAME=${2:-vanilla}  # Use the second argument if provided, otherwise default to "support"
    echo "Using folder: $FOLDER_NAME"
    isaaclab -p scripts/vla/sim-test.py
    exit 0
fi

if [ "$1" = "dev" ]; then
    echo "Running in dev mode"
    cd /workspace/c3po 
    /bin/bash
fi