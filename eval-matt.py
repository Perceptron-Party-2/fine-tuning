#%%
import transformers as t
import torch
import peft
import time

from bnb_config import bnb_config
from lora_config import lora_config
from prefix_config import prefix_config


peft_model_dir="./output/checkpoint-600"

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
# m = peft.get_peft_model(m, lora_config)
# m = peft.get_peft_model(m, prefix_config)
#peft.set_peft_model_state_dict(m, torch.load("./output/checkpoint-600/adapter_model.bin"))
# peft.set_peft_model_state_dict(m, torch.load("./output/checkpoint-600"))
"""""
config = peft.PeftConfig.from_pretrained(peft_model_dir)
# model = t.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
m = peft.PeftModel.from_pretrained(m, peft_model_dir)

"""
# m.to(device)
#%%
prompt = "Was ist die Hauptstadt Griechenlands? "#TEMPLATE.format(instruction=INSTRUCTION)
#%%
pipe = t.pipeline(task="text-generation", model=m, tokenizer=tokenizer, max_length = 100)
print("pipe(prompt)", pipe(prompt))