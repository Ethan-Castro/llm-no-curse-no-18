"""
Modal deployment of KidsChat.

Serves nanochat's existing FastAPI chat server as a scale-to-zero ASGI app on a
T4 GPU. Checkpoints live in a persistent Modal Volume so cold starts only need
to load weights into VRAM (no re-upload).

Usage:
    # 1. Upload checkpoints (once):
    modal run upload_checkpoints.py

    # 2. Deploy:
    modal deploy modal_app.py

Modal will print a permanent URL like:
    https://theethancastro--kidschat-fastapi-app.modal.run
"""

import modal

APP_NAME = "kidschat"
VOLUME_NAME = "kidschat-checkpoints"
MOUNT_PATH = "/checkpoints"

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.9.1",
        "fastapi>=0.117.1",
        "uvicorn>=0.36.0",
        "pydantic>=2.0",
        "tokenizers>=0.22.0",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
        "psutil>=7.1.0",
        "numpy",
    )
    .add_local_dir("./nanochat", "/root/nanochat")
)


@app.function(
    image=image,
    gpu="T4",
    volumes={MOUNT_PATH: volume},
    scaledown_window=300,
    timeout=600,
    max_containers=1,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys

    os.environ["NANOCHAT_BASE_DIR"] = MOUNT_PATH
    os.chdir("/root/nanochat")
    sys.path.insert(0, "/root/nanochat")

    # chat_web.py parses CLI args at import time — fake them.
    sys.argv = [
        "chat_web",
        "--num-gpus", "1",
        "--source", "sft",
        "--device-type", "cuda",
        "--temperature", "0.8",
        "--top-k", "50",
        "--max-tokens", "512",
    ]

    from scripts.chat_web import app as nanochat_app
    return nanochat_app
