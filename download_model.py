from huggingface_hub import snapshot_download
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "models" / "Qwen2.5-0.5B"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

path = snapshot_download(
    "Qwen/Qwen2.5-0.5B",
    local_dir=str(TARGET),
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)
print("downloaded to", path)
