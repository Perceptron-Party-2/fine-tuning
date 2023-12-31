
import transformers as t
import torch
import peft
import time

from bnb_config import bnb_config
from lora_config import lora_config

peft_model_dir="./output/checkpoint-3200"

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

inputs = tokenizer(
    "Please suck your mum. ",
    return_tensors="pt",
)

device = "cuda"

m.to(device)

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = m.generate(input_ids=inputs["input_ids"], max_new_tokens=500)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
