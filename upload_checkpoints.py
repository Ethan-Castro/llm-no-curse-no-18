"""
One-time upload of local ./checkpoints/ into the Modal Volume that modal_app.py
mounts at /checkpoints.

Usage:
    modal run upload_checkpoints.py

Re-run this whenever you retrain and want to push new weights.
"""

from pathlib import Path

import modal

VOLUME_NAME = "kidschat-checkpoints"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("kidschat-upload")


@app.local_entrypoint()
def main():
    local_root = Path(__file__).parent / "checkpoints"
    tokenizer_dir = local_root / "tokenizer"
    sft_dir = local_root / "chatsft_checkpoints"

    if not tokenizer_dir.is_dir():
        raise SystemExit(f"Missing tokenizer dir: {tokenizer_dir}")
    if not sft_dir.is_dir():
        raise SystemExit(f"Missing SFT checkpoints dir: {sft_dir}")

    print(f"Uploading checkpoints to Modal Volume '{VOLUME_NAME}'...")

    with volume.batch_upload(force=True) as batch:
        print(f"  + {tokenizer_dir} -> /tokenizer")
        batch.put_directory(str(tokenizer_dir), "/tokenizer")

        # Skip optimizer state (.pt files named optim_*) — not needed for inference.
        for sub in sorted(sft_dir.iterdir()):
            if not sub.is_dir():
                continue
            target = f"/chatsft_checkpoints/{sub.name}"
            print(f"  + {sub} -> {target}  (excluding optimizer state)")
            for f in sorted(sub.iterdir()):
                if f.name.startswith("optim_"):
                    continue
                batch.put_file(str(f), f"{target}/{f.name}")

    print("Upload complete.")
    print("Next: modal deploy modal_app.py")
