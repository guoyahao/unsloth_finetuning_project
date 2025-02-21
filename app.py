import gradio as gr
import yaml
from model_finetune import ModelFinetuner
from export_utils import export_to_gguf, export_to_onnx, push_to_huggingface

def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def finetune_and_export(model_name, dataset_name, max_seq_length, lora_r, lora_alpha, epochs, batch_size, learning_rate,
                        export_format, quantization_method, hf_repo, hf_token):
    """执行微调、导出和推送"""
    config = load_config("config.yaml")
    config.update({
        "model_name": model_name,
        "dataset_name": dataset_name,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "export_format": export_format,
        "quantization_method": quantization_method,
        "hf_repo": hf_repo
    })

    try:
        # 微调模型
        finetuner = ModelFinetuner(config)
        finetuner.load_model()
        finetuner.train()

        # 导出模型
        if export_format == "gguf":
            export_to_gguf(finetuner.model, finetuner.tokenizer, config["output_dir"], quantization_method)
        elif export_format == "onnx":
            export_to_onnx(finetuner.model, finetuner.tokenizer, config["output_dir"])

        # 推送至 Hugging Face
        if hf_repo and hf_token:
            push_to_huggingface(config["output_dir"], hf_repo, hf_token)

        return "Finetuning, export, and push completed successfully!"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio 界面
with gr.Blocks(title="Unsloth Model Finetuning") as demo:
    gr.Markdown("# Unsloth Model Finetuning Interface")
    gr.Markdown("Configure and finetune your model, then export and push to Hugging Face.")

    with gr.Row():
        with gr.Column():
            # 模型选择
            model_name = gr.Dropdown(
                label="Model Name",
                choices=["unsloth/llama-3-8b-bnb-4bit", "unsloth/mistral-7b-bnb-4bit", "unsloth/phi-4-GGUF"],
                value="unsloth/llama-3-8b-bnb-4bit"
            )
            dataset_name = gr.Textbox(label="Dataset Name", value="imdb")
            max_seq_length = gr.Slider(label="Max Sequence Length", minimum=512, maximum=8192, step=512, value=2048)
            lora_r = gr.Slider(label="LoRA Rank", minimum=4, maximum=64, step=4, value=16)
            lora_alpha = gr.Slider(label="LoRA Alpha", minimum=4, maximum=64, step=4, value=16)

        with gr.Column():
            epochs = gr.Slider(label="Epochs", minimum=1, maximum=10, step=1, value=1)
            batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, step=1, value=2)
            learning_rate = gr.Slider(label="Learning Rate", minimum=1e-5, maximum=1e-3, step=1e-5, value=2e-4)
            export_format = gr.Radio(label="Export Format", choices=["gguf", "onnx"], value="gguf")
            quantization_method = gr.Dropdown(label="Quantization Method (GGUF)", choices=["q4_k_m", "q8_0", "f16"], value="q4_k_m")

    with gr.Row():
        hf_repo = gr.Textbox(label="Hugging Face Repository", placeholder="your_username/your_model")
        hf_token = gr.Textbox(label="Hugging Face Token", type="password")

    run_button = gr.Button("Run Finetuning")
    output = gr.Textbox(label="Output")

    # 连接按钮与功能
    run_button.click(
        fn=finetune_and_export,
        inputs=[model_name, dataset_name, max_seq_length, lora_r, lora_alpha, epochs, batch_size, learning_rate,
                export_format, quantization_method, hf_repo, hf_token],
        outputs=output
    )

# 启动 Gradio 界面
demo.launch()