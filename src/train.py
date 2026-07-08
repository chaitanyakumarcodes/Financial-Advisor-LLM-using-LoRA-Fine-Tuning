from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

MODEL="meta-llama/Llama-3.2-3B-Instruct"

tokenizer=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")

def fmt(ex):
    text=f"""### Instruction
{ex["instruction"]}

### Input
{ex["input"]}

### Response
{ex["output"]}"""
    return {"text":text}

ds=load_dataset("json", data_files="../data/train.jsonl")["train"].map(fmt)

peft=LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)

args=TrainingArguments(
    output_dir="../outputs",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=1,
    save_strategy="epoch"
)

trainer=SFTTrainer(
    model=model,
    train_dataset=ds,
    args=args,
    peft_config=peft,
    formatting_func=lambda ex: ex["text"],
)

trainer.train()
trainer.save_model("../financial-advisor-lora")
