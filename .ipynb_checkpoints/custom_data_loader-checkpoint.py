import random
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from make_prompt import make_chat_prompt, get_code_completion  

instruction_prefix = (
    "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
)
response_prefix = (
    "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
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
            train_data = load_dataset('json', data_files="/project/phan/tqn/Adapter/LLM-SplitModel/data/train_data_qwen_100k.json")["train"]
            val_data = load_dataset('json', data_files="/project/phan/tqn/Adapter/LLM-SplitModel/data/val_data_qwen_100k.json")["train"]
        else:
            if self.data_path.endswith(".json") or self.data_path.endswith(".jsonl"):
                data = load_dataset("json", data_files=self.data_path)
            else:
                data = load_from_disk(self.data_path)
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42
            )
            if self.val_set_size > 0:
                train_data = train_val["train"].shuffle().map(self.generate_and_tokenize_prompt,batched=True, num_proc=128)
                val_data = train_val["test"].shuffle().map(self.generate_and_tokenize_prompt,batched=True, num_proc=128)
            else:
                train_data = data["train"].shuffle().map(self.generate_and_tokenize_prompt,batched=True, num_proc=128)
                val_data = None
            
            train_data.to_json("/project/phan/tqn/Adapter/LLM-SplitModel/data/train_data_qwen_eps27.json")
            if val_data:
                val_data.to_json("/project/phan/tqn/Adapter/LLM-SplitModel/data/val_data_qwen_eps27.json")

        return train_data, val_data


    # def generate_and_tokenize_prompt(self, batch):
    #     prompt_list = []
    #     input_ids_list = []
    #     attention_mask_list = []
    #     labels_list = []
    
    #     for instruction, test_list, code in zip(batch["text"], batch["test_list"], batch["code"]):
    #         # System prompt
    #         if test_list != "":
    #             system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    #         else:
    #             system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    
    #         # Combine instruction and test suite
    #         # instruction_with_tests = f"{instruction} Your code should pass these tests:\n\n{test_list}"
    
    #         # Generate full prompt
    #         full_prompt = self.prompter.generate_prompt(
    #             instruction,
    #             test_list[0],  # input_text not needed here
    #             "",
    #         )
    #         # Create chat-style messages
    #         messages = make_chat_prompt(
    #             full_prompt[len(system)+2:], instruction_prefix, response_prefix, self.tokenizer
    #         )
    #         tokenized_prompt = self.tokenize(messages)
    #         prompt_list.append(tokenized_prompt["input_ids"])
    #         # Append output/code
    #         if isinstance(code, list):
    #             messages += "\n".join(code)
    #         else:
    #             messages += code
    #         # Tokenize full prompt
    #         tokenized_full_prompt = self.tokenize(messages)
    
    #         # Mask input tokens if not training on inputs
    #         if not self.train_on_inputs:
    #             user_prompt = self.prompter.generate_prompt(instruction_with_tests, "")
    #             tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
    #             user_prompt_len = len(tokenized_user_prompt["input_ids"])
    #             tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    
    #         # Append to lists
    #         input_ids_list.append(tokenized_full_prompt["input_ids"])
    #         attention_mask_list.append(tokenized_full_prompt["attention_mask"])
    #         labels_list.append(tokenized_full_prompt["labels"])
    
    #     return {
    #         "prompt": prompt_list, 
    #         "input_ids": input_ids_list,
    #         "attention_mask": attention_mask_list,
    #         "labels": labels_list,
    #     }
    
    
    # def _load_dataset(self):
    #     # Load full MBPP dataset
    #     dataset = load_dataset("google-research-datasets/mbpp", "full")
    #     all_splits = []
    #     for split in dataset.keys():
    #         all_splits.append(dataset[split])
    
    #     # Concatenate all splits into a single dataset
    #     full_train_data = concatenate_datasets(all_splits)
    #     train_data = full_train_data.shuffle().map(self.generate_and_tokenize_prompt,batched=True, num_proc=128)
    #     val_data = full_train_data.shuffle().map(self.generate_and_tokenize_prompt,batched=True, num_proc=128)

    #     return train_data, val_data


