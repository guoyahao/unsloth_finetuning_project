from datasets import load_dataset

dataset = load_dataset("LooksJuicy/ruozhiba", "default", split="train", trust_remote_code=True)
print("列名:", dataset.column_names)