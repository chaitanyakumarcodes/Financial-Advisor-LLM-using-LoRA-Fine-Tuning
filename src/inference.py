from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL="meta-llama/Llama-3.2-3B-Instruct"
adapter="../financial-advisor-lora"

tok=AutoTokenizer.from_pretrained(MODEL)
base=AutoModelForCausalLM.from_pretrained(MODEL,device_map="auto")
model=PeftModel.from_pretrained(base,adapter)

prompt="I earn ₹80000/month. Help me create a budget."
inputs=tok(prompt,return_tensors="pt").to(model.device)
out=model.generate(**inputs,max_new_tokens=200)
print(tok.decode(out[0],skip_special_tokens=True))
