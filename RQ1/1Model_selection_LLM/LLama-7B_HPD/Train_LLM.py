import sys
import os
sys.path.append(os.getcwd()) 
from typing import List
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model 
from transformers import AutoTokenizer
from tqdm import tqdm
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel tokenization to avoid deadlocks.
from utils.prompter import Prompter

from datasets import Dataset
import random
from collections import Counter

def print_class_distribution(dataset):
    """
    Print the number of instances for each category in the dataset.

    Parameters:
    - dataset: A Dataset object containing the 'output' column.
    """
    # Count categories

    class_counts = Counter(dataset['output'])
    
    # Print each category and its count
    for class_label, count in class_counts.items():
        print(f"class {class_label}: {count} samples")

# Assuming you have a Dataset object named dataset
# Call the function to print category distribution
# print_class_distribution(dataset)


from collections import Counter
import random
from datasets import Dataset

from collections import Counter
import random
from datasets import Dataset

def balance_dataset_via_sampling(dataset, desired_balance_ratio=1.0, method="oversample"):
    """
    Balance the dataset categories based on a given ratio and specified method (oversampling or undersampling).

    Parameters:
    - dataset: A Dataset object containing the 'output' column.
    - desired_balance_ratio: The target ratio between the majority and minority classes.
    - method: The method to balance the dataset. "oversample" means oversampling the minority class, and "undersample" means undersampling the majority class.

    Returns:
    - The balanced Dataset object.
    """

    class_counts = Counter(dataset['output'])
    
    # 确定多数类和少数类
    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)
    
    # Calculate the target number based on the desired balance ratio
    if method == "oversample":
        # Oversample the minority class to the target ratio of the majority class count

        target_min_class_count = class_counts[max_class] / desired_balance_ratio
        if class_counts[min_class] < target_min_class_count:
            # Oversample the minority class
            min_class_indices = [i for i, output in enumerate(dataset['output']) if output == min_class]
            oversampled_min_indices = random.choices(min_class_indices, k=int(target_min_class_count))
            max_class_indices = [i for i, output in enumerate(dataset['output']) if output == max_class]
            balanced_indices = oversampled_min_indices + max_class_indices
    else:
        # Undersample the majority class to the target ratio of the minority class count
        target_max_class_count = class_counts[min_class] * desired_balance_ratio
        if class_counts[max_class] > target_max_class_count:
            # Undersample the majority class

            max_class_indices = [i for i, output in enumerate(dataset['output']) if output == max_class]
            undersampled_max_indices = random.sample(max_class_indices, k=int(target_max_class_count))
            min_class_indices = [i for i, output in enumerate(dataset['output']) if output == min_class]
            balanced_indices = undersampled_max_indices + min_class_indices
    
    # Create a new dataset
    balanced_dataset = dataset.select(balanced_indices)
    
    return balanced_dataset





def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,  # Truncate data that exceeds max_length
        max_length=cutoff_len,
        padding=False,  # No padding, use data_collator for padding during training
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()  # Labels for decoding models match the output
    return result

# Process the dataset
def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )  # Construct data with instruction template
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1
        # Set labels for non-training answers to -100, PyTorch ignores negative values in loss calculation
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt

# Load the base model
base_model =  "<Base_model_path>/bigscience/Vicuna-7B"
model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,  # Recommended data type for newer hardware (A100, A800, H800, RTX3090+). Use float16 for older hardware (V100, P40)
        device_map="auto",
)
AutoConfig.from_pretrained(base_model)
print("*"*40+"model load succ..."+"*"*40)


# 冻结原模型参数
list(model.parameters())[0].dtype
for i, param in enumerate(model.parameters()):
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)
model.gradient_checkpointing_enable()  
model.enable_input_require_grads()
class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

# Load LoRA model
lora_r = 8  # Width of the Lora matrix, increasing this value does not significantly improve performance according to the paper
lora_alpha = 16  # Original training alpha size for Lora, compressed by lora_r as described in the library
lora_target_modules = ["q_proj", "v_proj"]  # Names of attention model matrices affected by Lora, may vary by model; default for LLaMA
lora_dropout = 0.05  # Dropout size to prevent overfitting

config = LoraConfig(  # Lora settings
        r = lora_r,
        lora_alpha = lora_alpha,
        target_modules = lora_target_modules,  # target_modules must match the model structure
        lora_dropout = lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM",
    )

model = get_peft_model(model, config)  # Combine LoRA model with the base model
print("*" * 40 + "lora load succ..." + "*" * 40)
print_trainable_parameters(model)
print(model)

# Tokenization
print("*" * 40 + "tokenizing" + "*" * 40)
with tqdm(total=1, unit="models", desc="Loading tokenizer") as pbar:
    tokenizer = AutoTokenizer.from_pretrained(base_model)  # Initialize Tokenizer
    pbar.update(1)
print("*" * 40 + "finishing tokenizing" + "*" * 40)

model.get_input_embeddings()
tokenizer.pad_token_id = 0  # pad id, most padding ids are 0, but exceptions may exist
print("*" * 40 + "tokenizer load succ..." + "*" * 40)

output_dir: str = "./lora-checkpoint"  # Directory to store Lora model
cutoff_len: int = 256  # Truncation length, truncate tokens exceeding this length; must be less than model's token limit to avoid errors

# Apply Apache template for instruction fine-tuning
prompt_template_name: str = "alpaca"
prompter = Prompter(prompt_template_name)

# Additional parameters
train_on_inputs: bool = True  # Include model inputs in training
add_eos_token: bool = False  # Add end-of-sequence token to enhance model's stop output capability
group_by_length: bool = False  # Group data by token length. In a mini-batch, all data must complete inference before stopping, leading to computational waste with large length differences. Similar lengths might cause unstable training.

# Tokenize a single example

data_set = "HPD"
train_path = "./data/{}/train_set.json".format(data_set)  # Dataset path
test_path = "./data/{}/test_set.json".format(data_set)  # Dataset path
data_fields = {"train": train_path, "test": test_path}
data = load_dataset("json", data_files=data_fields)
balanced_dataset = data["train"]
val_set_size = int(len(data["train"]) * 0.2)

if val_set_size > 0:  # Split validation set
    train_val = balanced_dataset.train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

print("*" * 40 + "datastructing succ..." + "*" * 40)

micro_batch_size: int = 4  # Batch size for each computation, must divide batch_size evenly. For batch_size=128, forward pass 32 times before backward pass
num_epochs: int = 30  # Number of training epochs, i.e., how many times each data point is used
learning_rate: float = 3e-4  # Training learning rate, too high can prevent convergence, too low can slow convergence
batch_size = 128
gradient_accumulation_steps = batch_size // micro_batch_size  # Number of gradient accumulation steps

print("*" * 40 + "start training..." + "*" * 40)
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,  # Linear learning rate warm-up, see https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,  # Must match the model load setting, use fp16=True if torch.float16 was used during loading
        logging_steps=10,  # Log loss every 10 steps
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=5 if val_set_size > 0 else None,  # Evaluate every 5 steps
        save_steps=20,  # Save checkpoint every 20 steps
        output_dir=output_dir,
        save_total_limit=5,  # Maximum number of checkpoints to store
        load_best_model_at_end=True if val_set_size > 0 else False,
        group_by_length=group_by_length,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(  # Dynamic padding
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False
print("*"*40+"starting training..."+"*"*40)
trainer.train()
print("*"*40+"finishing training..."+"*"*40)


