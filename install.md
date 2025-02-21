# 环境安装部署
```bash
conda create --name unsloth_finetuning_project python=3.12 -y
conda activate unsloth_finetuning_project
```

## 1. PyTorch [https://pytorch.org/get-started/locally/]
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 2. 安装 Unsloth 
```bash
pip install --upgrade pip
pip install unsloth
```

## 3. 安装依赖
```bash
pip install onnx onnxruntime transformers
pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp312-cp312-win_amd64.whl
```
