#%%
import time
import transformers as t
import data
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import peft
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel, PeftConfig
import wandb
import hyper_params as h


is_ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if is_ddp else None

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#create 4bit model
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", quantization_config=bnb_config, device_map=device_map)

#configure lora
config = LoraConfig(
    r=h.LORA_R, 
    lora_alpha=h.LORA_ALPHA, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=h.LORA_DROPOUT, 
    bias="none", 
    task_type="CAUSAL_LM"
)

#add lora adaptors to model
model = get_peft_model(model, config)
#peft.set_peft_model_state_dict(m, torch.load("./output/checkpoint-600/adapter_model.bin"))
#peft.set_peft_model_state_dict(model, torch.load("./output/checkpoint-600/adapter_model.safetensors"))
model = PeftModel.from_pretrained(model, "./output/checkpoint-1200/")


#%%
TEMPLATE = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
INSTRUCTION = ""
prompt = "Hello what is your name?"#TEMPLATE.format(instruction=INSTRUCTION)
#%%
pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
print("pipe(prompt)", pipe(prompt))
