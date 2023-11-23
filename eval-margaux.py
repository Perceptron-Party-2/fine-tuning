
import transformers as t
import torch
import peft
import time

from bnb_config import bnb_config
from lora_config import lora_config

peft_model_dir="./genius/checkpoint-1600"

#%%
MODEL_NAME = "NousResearch/Llama-2-7b-hf"
tokenizer = t.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = 0

m = t.AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      quantization_config=bnb_config,
      use_cache=True,
      device_map= None
  )

m = peft.PeftModel.from_pretrained(m, peft_model_dir)

device = "cuda"
m.to(device)


TEMPLATE = "Below is something that a person has said to you. Write a response to that person.\n\n### Line:\n{line}\n\n### Response:\n"
LINE = "Why am I here then?"
prompt = TEMPLATE.format(line = LINE)
pipe = t.pipeline(task="text-generation", model=m, tokenizer=tokenizer, max_length=500)
print("pipe(prompt)", pipe(prompt))




