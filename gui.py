import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import yaml
import threading
import sys
import io
from model_finetune import ModelFinetuner
from export_utils import export_to_gguf, export_to_onnx, push_to_huggingface
from queue import Queue
import logging
import time
import pynvml

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("UnslothFinetuning")

class LogCapture(io.StringIO):
    def __init__(self, log_widget, queue):
        super().__init__()
        self.log_widget = log_widget
        self.queue = queue
        self.log_widget.tag_config("INFO", foreground="black")
        self.log_widget.tag_config("SUCCESS", foreground="green")
        self.log_widget.tag_config("WARNING", foreground="orange")
        self.log_widget.tag_config("ERROR", foreground="red")

    def write(self, message):
        self.queue.put(("log", message))

    def flush(self):
        pass

class FinetuneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unsloth 模型微调工具")
        self.config = self.load_config("config.yaml")
        self.log_queue = Queue()
        self.finetuner = None  # 保存微调后的模型和分词器
        self.create_widgets()
        self.root.geometry("1100x800")  # 调整窗口大小适应新布局
        self.running = False
        self.monitoring_gpu = False
        self.stop_event = threading.Event()
        self.check_queue()
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
        except pynvml.NVMLError:
            logger.warning("未检测到 NVIDIA GPU 或 NVML 初始化失败。\n")
            self.gpu_available = False

    def load_config(self, config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(config_file, "r", encoding="gbk") as f:
                return yaml.safe_load(f)

    def create_widgets(self):
        # 设置全局字体为微软雅黑，大小 11
        self.root.option_add("*Font", "微软雅黑 8")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 左侧：分为配置、导出和推送区域
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 上部：配置区域
        config_frame = ttk.LabelFrame(left_frame, text="配置")
        config_frame.pack(fill="both", expand=True, padx=5, pady=5)

        tk.Label(config_frame, text="模型名称:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar(value=self.config["model_name"])
        model_options = ["unsloth/llama-3-8b-bnb-4bit", "unsloth/mistral-7b-bnb-4bit", "unsloth/phi-4-GGUF"]
        tk.OptionMenu(config_frame, self.model_var, *model_options).grid(row=0, column=1)

        tk.Label(config_frame, text="数据集名称:").grid(row=1, column=0, padx=5, pady=5)
        self.dataset_var = tk.StringVar(value=self.config["dataset_name"])
        tk.Entry(config_frame, textvariable=self.dataset_var).grid(row=1, column=1)

        tk.Label(config_frame, text="最大序列长度:").grid(row=2, column=0, padx=5, pady=5)
        self.max_seq_var = tk.IntVar(value=self.config["max_seq_length"])
        tk.Entry(config_frame, textvariable=self.max_seq_var).grid(row=2, column=1)

        tk.Label(config_frame, text="LoRA 秩:").grid(row=3, column=0, padx=5, pady=5)
        self.lora_r_var = tk.IntVar(value=self.config["lora_r"])
        tk.Entry(config_frame, textvariable=self.lora_r_var).grid(row=3, column=1)

        tk.Label(config_frame, text="LoRA Alpha:").grid(row=4, column=0, padx=5, pady=5)
        self.lora_alpha_var = tk.IntVar(value=self.config["lora_alpha"])
        tk.Entry(config_frame, textvariable=self.lora_alpha_var).grid(row=4, column=1)

        tk.Label(config_frame, text="训练轮数:").grid(row=5, column=0, padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=self.config["epochs"])
        tk.Entry(config_frame, textvariable=self.epochs_var).grid(row=5, column=1)

        tk.Label(config_frame, text="批次大小:").grid(row=6, column=0, padx=5, pady=5)
        self.batch_var = tk.IntVar(value=self.config["batch_size"])
        tk.Entry(config_frame, textvariable=self.batch_var).grid(row=6, column=1)

        tk.Label(config_frame, text="学习率:").grid(row=7, column=0, padx=5, pady=5)
        self.lr_var = tk.DoubleVar(value=self.config["learning_rate"])
        tk.Entry(config_frame, textvariable=self.lr_var).grid(row=7, column=1)

        # 按钮区域（配置区域下部）
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=8, column=0, columnspan=2, pady=5)

        self.run_button = tk.Button(button_frame, text="开始微调", command=self.start_finetuning)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_frame, text="停止微调", command=self.stop_finetuning, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # 中部：导出区域（独立块，灰色背景）
        export_frame = ttk.LabelFrame(left_frame, text="导出", style="Custom.TLabelframe")
        export_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # 定义自定义样式为灰色背景
        style = ttk.Style()
        style.configure("Custom.TLabelframe", background="#f0f0f0")  # 浅灰色背景
        style.configure("Custom.TMenubutton", background="#f0f0f0")  # 下拉菜单背景

        self.export_gguf_button = tk.Button(export_frame, text="导出为 GGUF", command=self.export_to_gguf, bg="#f0f0f0")
        self.export_gguf_button.pack(pady=5, padx=10, fill=tk.X)

        self.export_onnx_button = tk.Button(export_frame, text="导出为 ONNX", command=self.export_to_onnx, bg="#f0f0f0")
        self.export_onnx_button.pack(pady=5, padx=10, fill=tk.X)

        tk.Label(export_frame, text="量化方法 (GGUF):", bg="#f0f0f0").pack(pady=5, padx=10)
        self.quant_var = tk.StringVar(value=self.config["quantization_method"])
        ttk.OptionMenu(export_frame, self.quant_var, "q4_k_m", "q8_0", "f16", style="Custom.TMenubutton").pack(pady=5, padx=10, fill=tk.X)

        # 下部：推送设置区域（独立块，灰色背景）
        push_frame = ttk.LabelFrame(left_frame, text="推送设置", style="Custom.TLabelframe")
        push_frame.pack(fill="both", expand=True, padx=5, pady=5)

        tk.Label(push_frame, text="Hugging Face 仓库:", bg="#f0f0f0").pack(pady=5, padx=10)
        self.hf_repo_var = tk.StringVar(value=self.config["hf_repo"])
        tk.Entry(push_frame, textvariable=self.hf_repo_var, bg="#f0f0f0").pack(pady=5, padx=10, fill=tk.X)

        tk.Label(push_frame, text="Hugging Face 令牌:", bg="#f0f0f0").pack(pady=5, padx=10)
        self.hf_token_var = tk.StringVar()
        tk.Entry(push_frame, textvariable=self.hf_token_var, show="*", bg="#f0f0f0").pack(pady=5, padx=10, fill=tk.X)

        self.push_hf_button = tk.Button(push_frame, text="推送到 Hugging Face", command=self.push_to_huggingface, bg="#f0f0f0")
        self.push_hf_button.pack(pady=5, padx=10, fill=tk.X)

        # 右侧：GPU 和日志区域
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        gpu_frame = ttk.LabelFrame(right_frame, text="GPU 负载")
        gpu_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.gpu_widget = scrolledtext.ScrolledText(gpu_frame, wrap=tk.WORD, height=10, bg="lightblue")
        self.gpu_widget.pack(fill="both", expand=True, padx=5, pady=5)

        log_frame = ttk.LabelFrame(right_frame, text="日志")
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_widget.pack(fill="both", expand=True, padx=5, pady=5)

        self.log_capture = LogCapture(self.log_widget, self.log_queue)
        sys.stdout = self.log_capture
        sys.stderr = self.log_capture

        # 设置权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)  # 配置
        left_frame.rowconfigure(1, weight=1)  # 导出
        left_frame.rowconfigure(2, weight=1)  # 推送
        right_frame.rowconfigure(0, weight=1)  # GPU 负载
        right_frame.rowconfigure(1, weight=1)  # 日志
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def check_queue(self):
        while not self.log_queue.empty():
            action, data = self.log_queue.get()
            if action == "log":
                self.update_log(data)
            elif action == "gpu":
                self.update_gpu(data)
            elif action == "finish":
                self.running = False
                self.monitoring_gpu = False
                self.run_button.config(state="normal")
                self.stop_button.config(state="disabled")
            elif action == "error":
                messagebox.showerror("错误", f"发生错误: {data}")
            elif action == "success":
                messagebox.showinfo("成功", "微调完成！")
        self.root.after(100, self.check_queue)

    def update_log(self, message):
        if "error" in message.lower():
            tag = "ERROR"
        elif "completed successfully" in message.lower() or "exported" in message.lower() or "pushed" in message.lower():
            tag = "SUCCESS"
        elif "warning" in message.lower() or "unsloth" in message.lower():
            tag = "WARNING"
        elif "stopped" in message.lower():
            tag = "WARNING"
        else:
            tag = "INFO"
        self.log_widget.insert(tk.END, message, tag)
        self.log_widget.see(tk.END)

    def update_gpu(self, message):
        self.gpu_widget.delete(1.0, tk.END)
        self.gpu_widget.insert(tk.END, message)

    def monitor_gpu(self):
        if not self.gpu_available:
            return
        device_count = pynvml.nvmlDeviceGetCount()
        while self.monitoring_gpu and not self.stop_event.is_set():
            gpu_info = ""
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info += f"GPU {i} ({name}):\n使用率: {util.gpu}%\n内存: {mem_info.used / 1024**2:.0f}MB / {mem_info.total / 1024**2:.0f}MB\n\n"
            self.log_queue.put(("gpu", gpu_info))
            time.sleep(2)

    def start_finetuning(self):
        if self.running:
            messagebox.showinfo("提示", "任务已在运行！")
            return
        self.running = True
        self.monitoring_gpu = True
        self.stop_event.clear()
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.log_widget.delete(1.0, tk.END)
        self.gpu_widget.delete(1.0, tk.END)
        self.finetune_thread = threading.Thread(target=self.run_finetuning)
        self.finetune_thread.start()
        self.gpu_thread = threading.Thread(target=self.monitor_gpu)
        self.gpu_thread.start()

    def stop_finetuning(self):
        if not self.running:
            messagebox.showinfo("提示", "没有任务在运行！")
            return
        self.stop_event.set()
        logger.info("微调被用户停止！\n")
        self.log_queue.put(("finish", None))

    def run_finetuning(self):
        try:
            self.config.update({
                "model_name": self.model_var.get(),
                "dataset_name": self.dataset_var.get(),
                "max_seq_length": self.max_seq_var.get(),
                "lora_r": self.lora_r_var.get(),
                "lora_alpha": self.lora_alpha_var.get(),
                "epochs": self.epochs_var.get(),
                "batch_size": self.batch_var.get(),
                "learning_rate": self.lr_var.get(),
                "quantization_method": self.quant_var.get(),
                "hf_repo": self.hf_repo_var.get()
            })

            logger.info("🦥 开始模型微调...\n")
            self.finetuner = ModelFinetuner(self.config)
            logger.info("🦥 Unsloth: 将修补您的计算机以启用2倍速免费微调。\n")
            logger.info("🦥 Unsloth Zoo 将优化一切以加速训练！\n")
            logger.info("加载模型...\n")
            self.finetuner.load_model()

            if self.stop_event.is_set():
                return

            logger.info("训练模型...\n")
            self.finetuner.train()

            if self.stop_event.is_set():
                return

            logger.info("微调成功完成！请选择导出或推送操作。\n")
            self.log_queue.put(("success", None))
            self.log_queue.put(("finish", None))
        except Exception as e:
            logger.error(f"发生错误: {str(e)}\n")
            self.log_queue.put(("error", str(e)))
            self.log_queue.put(("finish", None))

    def export_to_gguf(self):
        """导出为 GGUF 格式（随时可用）"""
        if not self.finetuner or not self.finetuner.model or not self.finetuner.tokenizer:
            messagebox.showerror("错误", "模型未加载！请先加载或微调模型。")
            return
        try:
            logger.info("正在导出为 GGUF...\n")
            export_to_gguf(self.finetuner.model, self.finetuner.tokenizer, self.config["output_dir"], self.quant_var.get())
            logger.info("导出为 GGUF 完成！\n")
        except Exception as e:
            logger.error(f"导出 GGUF 失败: {str(e)}\n")

    def export_to_onnx(self):
        """导出为 ONNX 格式（随时可用）"""
        if not self.finetuner or not self.finetuner.model or not self.finetuner.tokenizer:
            messagebox.showerror("错误", "模型未加载！请先加载或微调模型。")
            return
        try:
            logger.info("正在导出为 ONNX...\n")
            export_to_onnx(self.finetuner.model, self.finetuner.tokenizer, self.config["output_dir"])
            logger.info("导出为 ONNX 完成！\n")
        except Exception as e:
            logger.error(f"导出 ONNX 失败: {str(e)}\n")

    def push_to_huggingface(self):
        """推送到 Hugging Face"""
        if not self.finetuner or not self.finetuner.model or not self.finetuner.tokenizer:
            messagebox.showerror("错误", "模型未加载或微调未完成！")
            return
        if not self.hf_repo_var.get() or not self.hf_token_var.get():
            messagebox.showerror("错误", "请填写 Hugging Face 仓库和令牌！")
            return
        try:
            logger.info("正在推送到 Hugging Face...\n")
            push_to_huggingface(self.config["output_dir"], self.hf_repo_var.get(), self.hf_token_var.get())
            logger.info("推送至 Hugging Face 完成！\n")
        except Exception as e:
            logger.error(f"推送 Hugging Face 失败: {str(e)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = FinetuneGUI(root)
    root.mainloop()