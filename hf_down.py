import os
from huggingface_hub import snapshot_download

HUG = os.getenv('HUG')
downloaded_path = snapshot_download(
    local_dir='./',
    repo_id="dinleo11/Hecto",
    allow_patterns=["ckpt/*"],
    repo_type="model",
    token=HUG,
)

print(f"Downloaded to: {downloaded_path}")
