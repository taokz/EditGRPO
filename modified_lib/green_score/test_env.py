# test_vllm_install.py
try:
    import torch
    import transformers
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ Transformers: {transformers.__version__}")
    
    import vllm
    print(f"✓ vLLM imported successfully")
    
    from vllm import LLM, SamplingParams
    print("✓ vLLM classes imported successfully")
    
except ImportError as e:
    print(f"✗ Import error: {e}")