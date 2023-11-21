#%%
import transformers as t
import torch
import peft
import time

from bnb_config import bnb_config
from lora_config import lora_config
from prefix_config import prefix_config

#%%
MODEL_NAME = "NousResearch/Llama-2-7b-hf"
tokenizer = t.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = 0
m = t.AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      quantization_config=bnb_config,
      use_cache=False,
      device_map= None
  )
m = peft.get_peft_model(m, lora_config)
m = peft.get_peft_model(m, prefix_config)
#peft.set_peft_model_state_dict(m, torch.load("./output/checkpoint-600/adapter_model.bin"))
peft.set_peft_model_state_dict(m, torch.load("./output/checkpoint-600"))

#%%
TEMPLATE = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
INSTRUCTION = ""
prompt = "Hello what is your name?"#TEMPLATE.format(instruction=INSTRUCTION)
#%%
pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
print("pipe(prompt)", pipe(prompt))