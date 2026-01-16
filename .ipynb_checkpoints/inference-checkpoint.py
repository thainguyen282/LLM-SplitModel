import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from split_model.config import SplitConfig
from split_model.model import SplitModel, SplitModelForCausalLM
from utils.make_prompt import make_chat_prompt
from utils.prompter import Prompter
import os
import sys
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.append(repo_root)

instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"


def inference(
    model_path: str,
    prompt: str = "Write a python code to sum two numbers",
    prompt_template_name: str = "alpaca",
    max_new_tokens: int = 1024,
):
    """
    Run inference with a trained split model.
    
    Args:
        model_path: Path to the trained model checkpoint
        prompt: Input prompt for code generation
        prompt_template_name: Prompt template to use (default: "alpaca")
        max_new_tokens: Maximum number of tokens to generate
    """
    # Register model classes
    AutoConfig.register("split", SplitConfig)
    AutoModel.register(SplitConfig, SplitModel)
    AutoModelForCausalLM.register(SplitConfig, SplitModelForCausalLM)
    
    print(f"Loading model from: {model_path}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Prepare prompt
    prompter = Prompter(prompt_template_name)
    formatted_prompt = prompter.generate_prompt(prompt, "", "")
    chat_prompt = make_chat_prompt(formatted_prompt, instruction_prefix, response_prefix, tokenizer)
    
    print(f"\nInput prompt:\n{chat_prompt}\n")
    print("Generating response...\n")
    
    # Generate
    model_inputs = tokenizer([chat_prompt], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens
        )
        # Extract only the generated tokens (excluding the prompt)
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Generated output:\n{output_text}")


if __name__ == "__main__":
    fire.Fire(inference)
