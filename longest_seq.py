import json
from datasets import load_dataset

# Load the data using HuggingFace datasets
# Assumes the file is named 'train_data.json' and is in JSONL or JSON format
# If your file is JSONL, use data_files={'train': 'train_data.jsonl'}
data = load_dataset("json", data_files="train_data.json")['train']

# Find the longest instruction
max_len = 0
longest_instruction = ""
longest_instruction_idx = 0
for idx, item in enumerate(data):
    # Some datasets use "instruction", some use "input", adjust as needed
    instr = item.get("instruction", "")
    if len(instr) > max_len:
        max_len = len(instr)
        longest_instruction = instr
        longest_instruction_idx = idx
        
print("Longest instruction length:", max_len)
print(f"Longest instruction in line {longest_instruction_idx}:")
print(longest_instruction)
print("Row of longest instruction:")
print(data[longest_instruction_idx])

# Save the longest datapoint to train_data.json
longest_datapoint = data[longest_instruction_idx]
with open("train_data.json", "w") as f:
    json.dump(longest_datapoint, f, indent=2)

print(f"\nLongest datapoint saved to train_data.json")
