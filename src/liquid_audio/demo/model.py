"""Initialize models"""

import logging
import os

import torch

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

logger = logging.getLogger(__name__)

__all__ = ["device", "lfm2_audio", "mimi", "proc"]

HF_DIR = "LiquidAI/LFM2-Audio-1.5B"


def get_device() -> torch.device:
    """
    Detect and return the best available device.
    Priority: CUDA > Metal (macOS) > ROCm (AMD) > CPU
    
    Can be overridden with LIQUID_AUDIO_DEVICE environment variable.
    Valid values: "cuda", "mps", "cpu", "rocm" (or any valid torch device string)
    """
    device_env = os.environ.get("LIQUID_AUDIO_DEVICE", "").lower().strip()
    
    if device_env:
        try:
            device = torch.device(device_env)
            logger.info(f"Using device from LIQUID_AUDIO_DEVICE: {device}")
        except RuntimeError as e:
            logger.warning(f"Invalid device from LIQUID_AUDIO_DEVICE: {device_env}. Error: {e}")
        else:
            return device
    
    # Auto-detect best device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA available. Using device: {device}")
        return device
    
    # Check for Metal (macOS GPU)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            logger.info(f"Metal (MPS) available on macOS. Using device: {device}")
        except RuntimeError as e:
            logger.warning(f"Metal (MPS) available but failed to initialize: {e}")
        else:
            return device
    
    # Check for ROCm (AMD GPU)
    if hasattr(torch, "version") and hasattr(torch.version, "hip"):
        try:
            device = torch.device("cuda")  # ROCm uses CUDA API
            logger.info(f"ROCm (AMD GPU) available. Using device: {device}")
        except RuntimeError:
            pass
        else:
            return device
    
    # Fallback to CPU
    device = torch.device("cpu")
    logger.info(f"No GPU available. Using device: {device}")
    return device


device = get_device()

logging.info("Loading processor")
proc = LFM2AudioProcessor.from_pretrained(HF_DIR, device=device).eval()
logging.info("Loading model")
lfm2_audio = LFM2AudioModel.from_pretrained(HF_DIR, device=device).eval()
logging.info("Loading tokenizer")
mimi = proc.mimi.eval()

logging.info("Warmup tokenizer")
with mimi.streaming(1), torch.no_grad():
    for _ in range(5):
        x = torch.randint(2048, (1, 8, 1), device=device)
        mimi.decode(x)
