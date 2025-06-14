#!/bin/bash

echo "Starting entrypoint script..."
set -eo pipefail
shopt -s expand_aliases

cd /workspace/hand-rl
# pip install -e . --quiet

# cd /workspace
# # git clone https://github.com/svaichu/rldata.git
# cd rldata
# pip install -e . --quiet

# python -c "from opencv_fixer import AutoFix; AutoFix()"

echo "All dependencies installed successfully."

# if deploy arg is passed, run this file
# if [ "${1:-}" = "job" ]; then
#     python3 scripts/vla/groot-test.py
# fi

if [ "$1" = "dev" ]; then
    echo "Running in dev mode"
    cd /workspace/hand-rl
    /bin/bash
fi
