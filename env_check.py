import importlib, json, sys, platform

print("python", sys.version)
print("platform", platform.platform())

pkgs = ['torch', 'transformers', 'datasets', 'accelerate', 'numpy', 'matplotlib',
        'tqdm', 'safetensors', 'sentencepiece', 'triton', 'einops', 'huggingface_hub']
out = {}
for p in pkgs:
    try:
        m = importlib.import_module(p)
        out[p] = getattr(m, '__version__', '?')
    except Exception:
        out[p] = 'MISSING'
print(json.dumps(out, indent=1))

try:
    import torch
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        print("capability", torch.cuda.get_device_capability(0))
        print("cuda_version", torch.version.cuda)
        print("mem_GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    try:
        from torch.nn.attention.flex_attention import flex_attention
        print("flex_attention", "available")
    except Exception as e:
        print("flex_attention", "NO:", type(e).__name__, str(e)[:200])
    print("sdpa", hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
    print("flash_sdp", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp", torch.backends.cuda.mem_efficient_sdp_enabled())
except Exception as e:
    print("torch check failed:", e)
