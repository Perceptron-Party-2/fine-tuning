import transformers as t
import torch
import peft
import time

from bnb_config import bnb_config
from lora_config import lora_config

peft_model_dir="./chatbot_1/checkpoint-1800"

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


def eval(text, TEMPLATE):

    LINE = text
    prompt = TEMPLATE.format(line = LINE)
    pipe = t.pipeline(task="text-generation", model=m, tokenizer=tokenizer, max_length=500)
    #print("pipe(prompt)", pipe(prompt))
    return pipe(prompt)