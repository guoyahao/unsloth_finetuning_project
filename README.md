# Unsloth Model Finetuning GUI

这是一个使用 Unsloth 进行模型微调的图形界面工具，支持模型选择、参数调整、导出格式选择（ONNX 和 GGUF）以及推送至 Hugging Face。

## 安装
1. 克隆项目：
   ```bash
   git clone https://github.com/guoyahao/unsloth_finetuning_project
   cd unsloth_finetuning_project
   conda activate unsloth_finetuning_project
   ```   


2. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```
## 使用GUI
1. 运行 GUI：
    ```bash
    python gui.py
   ```
2. 在界面中选择模型、调整参数、选择导出格式并输入 Hugging Face 信息。
3. 点击 "Run Finetuning" 开始微调和导出。

## 使用Gradio界面
1. 运行程序：
    ```bash
    python app.py
   ```
2. 在浏览器中打开 Gradio 界面（通常是 http://127.0.0.1:7860）。
3. 配置模型参数、选择导出格式并输入 Hugging Face 信息。
4. 点击 "Run Finetuning" 执行微调、导出和推送。

## 注意事项
1. 确保 GPU 环境正确配置（支持 CUDA）。
2. 需要 Hugging Face 账户和有效的 Token 用于推送模型。
3. 默认配置文件 config.yaml 可作为参考。
4. Gradio 界面支持远程访问，可通过 demo.launch(share=True) 启用公网链接。

---

### 使用说明
1. **运行程序**：
   - 执行 `python gui.py` 启动图形界面。
   - 或执行 `python app.py` 启动 Gradio 界面。
2. **模型选择**：
   - 从下拉菜单选择支持的模型（如 LLaMA、Mistral 等）。
3. **参数调整**：
   - 输入数据集名称、LoRA 参数、训练超参数等。
4. **导出格式**：
   - 选择导出为 GGUF 或 ONNX，并为 GGUF 指定量化方法。
5. **推送至 Hugging Face**：
   - 输入仓库名称和 Token，完成后自动推送。
6. **执行微调**：
   - 点击 "Run Finetuning" 按钮，程序将依次完成微调、导出和推送。

---

### 注意事项
- **环境要求**：确保安装了支持 CUDA 的 PyTorch 和相关依赖。
- **Hugging Face Token**：需要在 Hugging Face 官网生成 Token 并在 GUI 中输入。
- **数据集**：默认使用 IMDb 数据集，可替换为自定义数据集（需确保格式兼容）。
- **资源需求**：微调和导出需要足够的 GPU 内存（建议至少 12GB）。
