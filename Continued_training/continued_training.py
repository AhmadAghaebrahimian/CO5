# -*- coding: utf-8 -*-
"""
Task LLM Continued Training Solution

Created on Wed November 19 07:51:46 2025

@author: agha
"""

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

max_seq_length = 1024
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2b-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    bias="none",
    lora_alpha = 16,
    lora_dropout = 0,
    random_state=1234,
    use_rslora=False,
    loftq_config=None,
    use_gradient_checkpointing = "unsloth",
)


lm_prompt = """{}"""
EOS_TOKEN = tokenizer.eos_token # each sequence should be ended with this special character, so LLM learns when to stop

def format_dataset_lm():

    texts = [] # list of string each consists of CHARACTER: Utterance formatted in lm_prompt.eg. """MARCIUS: They have a leader,"""

    return Dataset.from_dict({"text": texts, })

dataset = format_dataset_lm()


# Config the trainer
sftConfig = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    # max_steps = 1,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=1234,
    output_dir="outputs",
)


# Trainer object
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = sftConfig
)

t_obj = trainer.train()

# Switch to inference mode
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    lm_prompt.format("HAMLET: ")
], return_tensors = "pt").to("cuda")


text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
