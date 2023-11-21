from peft import TaskType, LoraConfig

from hyper_params import LORA_R, LORA_ALPHA, LORA_DROPOUT

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)