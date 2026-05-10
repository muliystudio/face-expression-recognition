import sys
import torch
import PySide6

# 1. 查看 Python 版本
print(f"Python 版本: {sys.version}")

# 2. 查看 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 3. 查看 CUDA 信息 (通过 PyTorch 检测)
if torch.cuda.is_available():
    print(f"CUDA 是否可用: True")
    print(f"PyTorch 编译对应的 CUDA 版本: {torch.version.cuda}")
    print(f"当前 GPU 设备数量: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA 是否可用: False (未检测到 GPU 或 PyTorch 为 CPU 版本)")

# 4. 查看 PySide6 版本
print(f"PySide6 版本: {PySide6.__version__}")