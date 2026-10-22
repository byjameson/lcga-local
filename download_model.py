from huggingface_hub import snapshot_download
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
path = snapshot_download(
    "Qwen/Qwen2.5-0.5B",
    local_dir=r"C:\Users\biris\Desktop\opencode\ai gelistir\models\Qwen2.5-0.5B",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)
print("downloaded to", path)
