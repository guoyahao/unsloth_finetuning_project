from unsloth import FastLanguageModel
from optimum.onnxruntime import ORTModelForCausalLM
from huggingface_hub import HfApi, login

def export_to_gguf(model, tokenizer, output_dir, quantization_method="q4_k_m"):
    """导出模型到 GGUF 格式"""
    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization_method)
    print(f"Model exported to GGUF at {output_dir} with quantization {quantization_method}")

def export_to_onnx(model, tokenizer, output_dir):
    """导出模型到 ONNX 格式"""
    ort_model = ORTModelForCausalLM.from_pretrained(output_dir, export=True)
    ort_model.save_pretrained(output_dir + "_onnx")
    tokenizer.save_pretrained(output_dir + "_onnx")
    print(f"Model exported to ONNX at {output_dir}_onnx")

def push_to_huggingface(output_dir, repo_name, token):
    """推送模型到 Hugging Face"""
    login(token)
    api = HfApi()
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_name,
        repo_type="model"
    )
    print(f"Model pushed to Hugging Face at {repo_name}")