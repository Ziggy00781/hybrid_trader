import torch
import os

def get_device():
    # Force CPU if user sets this env var
    if os.getenv("FORCE_CPU", "0") == "1":
        return "cpu"

    # If CUDA is available, use it
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"