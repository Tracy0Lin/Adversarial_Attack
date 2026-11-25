# Start the Flask app with the CUDA-enabled VLAttack env.
# Do not use C:\Python310\python.exe — that interpreter has no torch/CUDA.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$gpuPython = "C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_zip\vlattack_gpu\Scripts\python.exe"
if (-not (Test-Path $gpuPython)) {
    throw "找不到 GPU Python：$gpuPython"
}
$env:VLATTACK_DEVICE = "cuda"
Set-Location $root
Write-Host "Using $gpuPython"
& $gpuPython -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.cuda.get_device_name(0))"
& $gpuPython app.py
