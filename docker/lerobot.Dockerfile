FROM huggingface/lerobot-gpu:latest

RUN bash -c "pip install -e '.[hilserl]'"

WORKDIR /workspace/hand-rl