import random
import os
from datasets import load_dataset, Dataset, load_from_disk
from utils.make_prompt import make_chat_prompt, get_code_completion  

instruction_prefix = (
    "Think step by step: please provide an efficient and self-contained Python script "
    "that solves the following problem in a markdown code block:"
)
response_prefix = (
    "Below is a Python script with a self-contained function that efficiently solves "
    "the problem and passes corresponding tests:"
)


class LoadData:
    def __init__(
        self,
        data_path,
        tokenizer,
        prompter,
        cutoff_len,
        train_on_inputs,
        using_raw_data = True, 
        val_set_size=2000
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.cutoff_len = cutoff_len
        self.train_on_inputs = train_on_inputs
        self.using_raw_data = using_raw_data
        
        self.val_set_size = val_set_size
        self.template = '    """{}"""\n'
        self.train_data, self.val_data = self._load_dataset()

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt_auto_completion(self, data_point):
        if data_point["input"] != "":
            system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        else:
            system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

        temp_prompt = data_point["output"].split(":")[0] + ":\n" + self.template.format(data_point["instruction"])
        
        output = data_point["output"]

        temp_tokenizer = self.tokenizer(
            output,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )

        if len(temp_tokenizer["input_ids"]) == 1:
            full_prompt = self.prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                "",
            )
            messages = make_chat_prompt(
                full_prompt[len(system)+2:], instruction_prefix, response_prefix, self.tokenizer
            ) + output
        else:
            random_numbers = sorted(random.sample(range(0, len(temp_tokenizer["input_ids"])), k=2))
            prefix = self.tokenizer.batch_decode(
                [temp_tokenizer["input_ids"][:random_numbers[0]]],
                skip_special_tokens=False
            )[0] if random_numbers[0] > 0 else self.tokenizer.batch_decode(
                [temp_tokenizer["input_ids"][random_numbers[0]]],
                skip_special_tokens=False
            )[0]

            suffix = (
                "\"\"\""
                if random_numbers[1] == len(temp_tokenizer["input_ids"])
                else self.tokenizer.batch_decode(
                    [temp_tokenizer["input_ids"][random_numbers[1]:]],
                    skip_special_tokens=False
                )[0]
            )

            temp_prompt = get_code_completion(prefix, suffix)
            temp_prompt = self.prompter.generate_prompt(
                temp_prompt,
                data_point["input"],
                "",
            )

            messages = make_chat_prompt(temp_prompt, instruction_prefix, response_prefix, self.tokenizer)
            middle = self.tokenizer.batch_decode(
                [temp_tokenizer["input_ids"][random_numbers[0]:random_numbers[1]]],
                skip_special_tokens=False
            )[0]
            messages += middle

        tokenized_full_prompt = self.tokenize(messages)

        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    def generate_and_tokenize_prompt(self, batch):

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for instruction, input_text, output in zip(batch["instruction"], batch["input"], batch["output"]):
            if input_text != "":
                system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
            else:
                system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    
            full_prompt = self.prompter.generate_prompt(
                instruction,
                input_text,
                "",
            )
    
            messages = make_chat_prompt(
                full_prompt[len(system)+2:], instruction_prefix, response_prefix, self.tokenizer
            )

            if isinstance(output,list):
                messages += "".join(output)
            else:
                messages += output

            #print(messages)
            #exit()
            tokenized_full_prompt = self.tokenize(messages)
    
            if not self.train_on_inputs:
                user_prompt = self.prompter.generate_prompt(
                    instruction, input_text
                )
                tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
    
                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
                
            # Append tokenized data to respective lists
            input_ids_list.append(tokenized_full_prompt["input_ids"])
            attention_mask_list.append(tokenized_full_prompt["attention_mask"])
            labels_list.append(tokenized_full_prompt["labels"])
    
            #return tokenized_full_prompt
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    def _load_dataset(self):
        if not self.using_raw_data:
            # Determine directory for preprocessed data files
            if self.data_path.endswith(".json") or self.data_path.endswith(".jsonl"):
                data_dir = os.path.dirname(self.data_path) if os.path.dirname(self.data_path) else "."
            else:
                data_dir = self.data_path
            
            # Construct paths for preprocessed train and validation data
            train_data_path = os.path.join(data_dir, "train_data_preprocessed.json")
            val_data_path = os.path.join(data_dir, "val_data_preprocessed.json")
            
            train_data = load_dataset('json', data_files=train_data_path)["train"]
            val_data = load_dataset('json', data_files=val_data_path)["train"]
        else:
            if self.data_path.endswith(".json") or self.data_path.endswith(".jsonl"):
                data = load_dataset("json", data_files=self.data_path)
                data_dir = os.path.dirname(self.data_path) if os.path.dirname(self.data_path) else "."
            else:
                data = load_from_disk(self.data_path)
                data_dir = self.data_path
            
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42
            )
            if self.val_set_size > 0:
                train_data = train_val["train"].shuffle().map(self.generate_and_tokenize_prompt,batched=True, num_proc=128)
                val_data = train_val["test"].shuffle().map(self.generate_and_tokenize_prompt,batched=True, num_proc=128)
            else:
                train_data = data["train"].shuffle().map(self.generate_and_tokenize_prompt,batched=True, num_proc=128)
                val_data = None
            
            # Save preprocessed data to the same directory as the input data
            train_data_path = os.path.join(data_dir, "train_data_preprocessed.json")
            val_data_path = os.path.join(data_dir, "val_data_preprocessed.json")
            
            train_data.to_json(train_data_path)
            if val_data:
                val_data.to_json(val_data_path)

        return train_data, val_data


       
