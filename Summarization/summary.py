# -*- coding: utf-8 -*-
"""
Task LLM Continued Training Solution

Created on Wed November 19 10:45:11 2025

@author: agha
"""

from unsloth import FastLanguageModel
import os
import json
import torch
from tqdm import tqdm
from bert_score import score
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

max_seq_length = 7000
load_in_4bit = True

sum_pubmed_base_path = "/home/ahmad/Resources/SumPubMed/sumpubmed-master"
texts_path = os.path.join(sum_pubmed_base_path, 'abstract')# training on full text takes too much time
shorter_abstract_path = os.path.join(sum_pubmed_base_path, 'shorter_abstract')

alpaca_instruction = """Below is an instruction that describes a task, paired with an input that provide further context. Write a response that appropriately completes the request.

### Instruction:
Your task is to generate a summary for a medical transcript in a clear, concise, and meaningful way, ensuring that essential information is captured without unnecessary details. This should be presented in less than 300 words. Here’s a step-by-step guide to help you approach this professionally. 1. Understand the Purpose. 2. Focus on Key Elements. 3. Be Clear and Concise. 4. Use Structured Writing. 5. Review for Accuracy. Make sure that all important medical details are captured in the summary without distorting meaning. Double-check for clarity, grammar, and brevity. \  

### input:
{}

### Response:
{}"""

# huggingface-cli download unsloth/Llama-3.2-3B-Instruct-bnb-4bit --local-dir ./models/Llama-3.2-3B-Instruct-bnb-4bit
# base_model = "./models/gemma-2b-bnb-4bit"#"unsloth/llama-3-8b-bnb-4bit"
base_model = "./models/Llama-3.2-3B-Instruct-bnb-4bit"

def fine_tune(step):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        load_in_4bit = load_in_4bit, )

    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=16,

        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1234,
        use_rslora=False,
        loftq_config=None,
    )


    def formatting_training_lines_sumpubmed():
        EOS_TOKEN = tokenizer.eos_token
        texts = []

        for i in range(1, 100):
            text = open(os.path.join(texts_path, 'abstract_%d.txt'%i), 'r').read()[:max_seq_length]
            summary = open(os.path.join(shorter_abstract_path, 'abst_%d.txt'%i), 'r').read()[:max_seq_length]

            full_text = alpaca_instruction.format(text, summary) + EOS_TOKEN

            texts.append(full_text)

        return Dataset.from_dict({"text": texts, })
    dataset = formatting_training_lines_sumpubmed()

    sftConfig = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=step,
        # num_train_epochs=epochs,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.03,
        lr_scheduler_type="linear",
        seed=1234,
        output_dir="outputs%s" % local_model_name,
    )

    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=sftConfig
    )

    trainer.train()

    FastLanguageModel.for_inference(peft_model)

    titles = []
    for i in tqdm(range(1, 10)):
        text = open(os.path.join(texts_path, 'abstract_%d.txt' % i), 'r').read()[:max_seq_length]
        summary = open(os.path.join(shorter_abstract_path, 'abst_%d.txt' % i), 'r').read()[:max_seq_length]

        in_text = alpaca_instruction.format(text, "")
        inputs = tokenizer([in_text], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=250, use_cache=True)
        out_text = tokenizer.batch_decode(outputs)[0]
        pred_title = out_text.split("### Summary:")[-1].replace('<|end_of_text|>', '').strip()
        titles.append({'true':summary, 'pred':pred_title})

    with open(write_to, 'w') as wr:
        json.dump(titles, wr, indent=1)

    P, R, F1 = score([x['pred'] for x in titles], [x['true'] for x in titles], lang="en", verbose=True)
    print("Stats after finetuning ...")
    print("Average Precision:", P.mean().item())
    print("Average Recall:", R.mean().item())
    print("Average F1:", F1.mean().item())


def raw_model_test():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        load_in_4bit = load_in_4bit, )

    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1234,
        use_rslora=False,
        loftq_config=None,
    )

    FastLanguageModel.for_inference(peft_model)

    titles = []

    for i in tqdm(range(1, 10)):
        text = open(os.path.join(texts_path, 'abstract_%d.txt' % i), 'r').read()[:max_seq_length]
        summary = open(os.path.join(shorter_abstract_path, 'abst_%d.txt' % i), 'r').read()[:max_seq_length]

        in_text = alpaca_instruction.format(text, "")
        inputs = tokenizer([in_text], return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=250, use_cache=True)
        out_text = tokenizer.batch_decode(outputs)[0]

        pred_title = out_text.split("### Response:")[-1].replace('<|end_of_text|>', '').strip()
        titles.append({'true':summary, 'pred':pred_title})


    with open(write_to, 'w') as wr:
        json.dump(titles, wr, indent=1)

    P, R, F1 = score([x['pred'] for x in titles], [x['true'] for x in titles], lang="en", verbose=True)
    print("Stats without finetuning ..." )
    print("Average Precision:", P.mean().item())
    print("Average Recall:", R.mean().item())
    print("Average F1:", F1.mean().item())



if __name__ == "__main__":
    step = 1
    write_to = 'sumPubMed.json'
    local_model_name = 'pubmed_title_step%d' % (step)
    fine_tune(step)

# Stats without finetuning ...
# Average Precision: 0.9236090779304504
# Average Recall: 0.8756392002105713
# Average F1: 0.8987348079681396

# Stats after finetuning(step 3) ...
# Average Precision: 0.9357865452766418
# Average Recall: 0.8789170384407043
# Average F1: 0.9060934782028198

# Stats after finetuning(step 5) ...
# Average Precision: 0.9408960342407227
# Average Recall: 0.9042227268218994
# Average F1: 0.9220747351646423