# Device Configuration Guide

The Liquid Audio demo now supports running on multiple device types: NVIDIA CUDA GPUs, Apple Metal (macOS), AMD GPUs (ROCm), and CPU.

## Automatic Device Detection

By default, the demo automatically detects and uses the best available device in this priority order:

1. **NVIDIA CUDA GPU** - If CUDA is available
2. **Apple Metal (macOS)** - If running on macOS with Metal support
3. **AMD GPU (ROCm)** - If ROCm is available
4. **CPU** - Fallback option

## Manual Device Selection

You can override the automatic detection by setting the `LIQUID_AUDIO_DEVICE` environment variable:

### Using NVIDIA CUDA GPU
```bash
# Linux/Mac
export LIQUID_AUDIO_DEVICE=cuda
liquid-audio-demo

# Windows (PowerShell)
$env:LIQUID_AUDIO_DEVICE="cuda"
liquid-audio-demo

# Windows (Command Prompt)
set LIQUID_AUDIO_DEVICE=cuda
liquid-audio-demo
```

### Using Apple Metal (macOS)
```bash
# macOS with Apple Silicon (M1, M2, M3, M4, etc.)
export LIQUID_AUDIO_DEVICE=mps
liquid-audio-demo
```

**Note:** Metal support requires:
- macOS 12.3 or later
- PyTorch with MPS support: `pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

### Using AMD GPU (ROCm)
```bash
# Linux/Mac
export LIQUID_AUDIO_DEVICE=cuda
liquid-audio-demo

# Windows (PowerShell)
$env:LIQUID_AUDIO_DEVICE="cuda"
liquid-audio-demo

# Windows (Command Prompt)
set LIQUID_AUDIO_DEVICE=cuda
liquid-audio-demo
```

**Note:** AMD ROCm also uses the `cuda` device string in PyTorch. Make sure you have ROCm properly installed.

### Using CPU
```bash
# Linux/Mac
export LIQUID_AUDIO_DEVICE=cpu
liquid-audio-demo

# Windows (PowerShell)
$env:LIQUID_AUDIO_DEVICE="cpu"
liquid-audio-demo

# Windows (Command Prompt)
set LIQUID_AUDIO_DEVICE=cpu
liquid-audio-demo
```

### Using Specific GPU Index
If you have multiple GPUs and want to use a specific one:
```bash
# Linux/Mac
export LIQUID_AUDIO_DEVICE=cuda:0  # First GPU
export LIQUID_AUDIO_DEVICE=cuda:1  # Second GPU

# Windows (PowerShell)
$env:LIQUID_AUDIO_DEVICE="cuda:0"
liquid-audio-demo
```

## Performance Considerations

### NVIDIA CUDA
- **Best performance** for most models
- Requires NVIDIA GPU with CUDA Compute Capability 3.5+
- Install: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Apple Metal (macOS)
- **Excellent performance** on Apple Silicon (M1/M2/M3/M4)
- Native GPU acceleration with unified memory
- Requires macOS 12.3+ and nightly PyTorch build
- Install: `pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

### AMD ROCm
- **Good performance** on AMD GPUs
- Requires AMD GPU with RDNA or CDNA architecture
- Limited support on newer iGPUs (890M not yet supported)
- Install: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7`

### CPU
- **Slowest option**, suitable for testing and development
- Works on any machine
- No special installation required

## Troubleshooting

### Device not detected
If your device isn't being detected automatically, check:
1. PyTorch is installed for your device type
2. GPU drivers are properly installed
3. Try setting `LIQUID_AUDIO_DEVICE` manually

### Out of Memory (OOM) errors
- On GPU: Reduce batch size or use CPU
- On CPU: Ensure sufficient RAM (model requires ~6GB)

### Metal not working on macOS
- Verify Metal support: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Ensure you have the nightly PyTorch build: `pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`
- Check Xcode command-line tools: `xcode-select --install`
- Metal is only available on Apple Silicon (M1/M2/M3/M4) and some AMD Macs

### ROCm not working
- Verify ROCm installation: `python -c "import torch; print(torch.version.hip)"`
- Check GPU support: `rocm-smi`
- Reinstall PyTorch for ROCm: `pip install torch --index-url https://download.pytorch.org/whl/rocm5.7`
- Note: Radeon 890M (gfx1150) is not yet officially supported in PyTorch ROCm

## Checking Your Device

To verify which device is being used, check the logs when starting the demo:
```
INFO:liquid_audio.demo.model:CUDA available. Using device: cuda
# or
INFO:liquid_audio.demo.model:Metal (MPS) available on macOS. Using device: mps
# or
INFO:liquid_audio.demo.model:No GPU available. Using device: cpu
```
