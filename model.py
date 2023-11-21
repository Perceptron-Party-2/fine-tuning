# %%

import transformers as t
import peft
import torch
import os

from bnb_config import bnb_config
from lora_config import lora_config
from prefix_config import prefix_config

def get_model():
  MODEL_NAME = "NousResearch/Llama-2-7b-hf"
  is_ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
  device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if is_ddp else None
  m = t.AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      quantization_config=bnb_config,
      use_cache=False,
      device_map=device_map
  )
  m = peft.prepare_model_for_kbit_training(m)
  m = peft.get_peft_model(m, lora_config)
  m = peft.get_peft_model(m, prefix_config)
  
  return m

model = get_model()
print(model.parameters())
# %%
