import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from utils.prompter import Prompter
from utils.Resultsaver import Results
from datasets import load_dataset

from tqdm import tqdm
# sys.path.append(os.getcwd())



@torch.inference_mode()
def evaluate(
    model,
    instruction,
    input=None,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=256,
    max_input = 256,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input, padding= True)
    input_ids = inputs["input_ids"].to(device)

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            top_p=top_p,  # Different inference parameters will yield different outputs
            top_k=top_k,  # See https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/text_generation#transformers.GenerationConfig for details
            num_beams=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)  # Convert generated ids to string
    return prompter.get_response(output)  # Output only the newly generated result


   



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

base_model =  "<Base_model_path>/bigscience/Vicuna-13B"
lora_weights= "./lora-checkpoint/checkpoint-1140"
prompt_template = "alpaca"  # Template to use, located in the templates folder
base_model = base_model or os.environ.get("BASE_MODEL", "")
assert base_model, "Please specify a --base_model, e.g. --base_model='bigscience/Vicuna-13B'"

prompter = Prompter(prompt_template)
tokenizer = AutoTokenizer.from_pretrained(base_model)

if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,  # Data type for the model; bfloat16 recommended for newer hardware (A100, A800, H800, RTX3090+), float16 for older hardware (V100, P40)
        device_map="auto",
    )
    lora_model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model, low_cpu_mem_usage=True
    )
    lora_model = PeftModel.from_pretrained(
        model,
        lora_weights,
    )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # pad id, most padding ids are 0, but exceptions may exist
    lora_model.config.pad_token_id = 0

    model.eval()

    data_set = "HPD"
    train_path = "./data/{}/train_set.json".format(data_set)  # Dataset path
    test_path = "./data/{}/test_set.json".format(data_set)  # Dataset path
    data_fields = {"train": train_path, "test": test_path}
    data = load_dataset("json", data_files=data_fields)

    # Load test set
    test_data = data["test"]
    print("*" * 40 + "test results..." + "*" * 40)

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)  # PyTorch 2.0 and later versions can compile to significantly improve training efficiency


down = Results(hyperParas="lora",approach=lora_weights.replace("./","").replace("/","-"),dataSets=data_set)
iter = 0
total_items = len(test_data) 
for data in tqdm(test_data, total=total_items, desc="Testing"):
    iter += 1
    instruction = data['instruction']
    input = data['input']
    labels = data['output']
    x = input
    groundtruth = labels
    # if groundtruth == "No":continue
    
    
    log = "Input:"+x+"\n"
    log += "labels:"+groundtruth+"\n"
    
    try:
        lorapred = evaluate(lora_model,instruction,input=input)
    except:
        continue
    log += "Result:"+lorapred+"\n"
    down.savelogDData(log)
    
    log = "Iter{} Finished".format(iter)+"\n\n"
    down.savelogDData(log)