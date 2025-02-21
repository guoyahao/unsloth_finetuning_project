from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch

# 默认提示模板（可通过 config 自定义）
DEFAULT_PROMPT_STYLE = "问题: {}\n复杂推理: {}\n回答: {}"


class ModelFinetuner:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.eos_token = None

    def load_model(self):
        """加载模型和分词器，并获取 EOS_TOKEN"""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config["model_name"],
            max_seq_length=self.config["max_seq_length"],
            dtype=None,
            load_in_4bit=True
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            use_rslora= False,
            loftq_config= None,
        )
        self.eos_token = self.tokenizer.eos_token  # 获取 EOS_TOKEN

    def formatting_prompts_func(self, examples):
        input_fields = self.config.get("input_fields", ["instruction", "output"])
        prompt_style = self.config.get("prompt_style", "指令: {}\n回答: {}")

        available_fields = examples.keys()
        missing_fields = [f for f in input_fields if f not in available_fields]
        if missing_fields:
            raise ValueError(f"数据集缺少字段: {missing_fields}")

        field_values = [examples[field] for field in input_fields]
        texts = []
        for values in zip(*field_values):
            text = prompt_style.format(*values) + self.eos_token
            texts.append(text)
        return {"text": texts}

    def load_dataset(self):
        dataset = load_dataset(
            self.config["dataset_name"],
            self.config.get("dataset_config_name", "default"),  # 默认使用 'default'
            split=self.config.get("dataset_split", "train"),
            num_proc=1,  # 强制单进程
            trust_remote_code=True
        )
        dataset = dataset.map(self.formatting_prompts_func, batched=True)
        return dataset

    def train(self):
        """训练模型"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not loaded!")

        dataset = self.load_dataset()
        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            learning_rate=self.config["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            save_strategy="epoch"
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",  # 使用格式化后的 'text' 字段
            max_seq_length=self.config["max_seq_length"],
            dataset_num_proc=1,
            packing = False,
            args=training_args
        )

        trainer.train()
        self.model.save_pretrained(self.config["output_dir"])
        self.tokenizer.save_pretrained(self.config["output_dir"])