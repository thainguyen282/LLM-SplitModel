import random
from datasets import load_dataset, Dataset, load_from_disk
from make_prompt import make_chat_prompt, get_code_completion  

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

    def generate_and_tokenize_prompt(self, data_point):
        if data_point["input"] != "":
            system = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        else:
            system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            "",
        )

        messages = make_chat_prompt(
            full_prompt[len(system)+2:], instruction_prefix, response_prefix, self.tokenizer
        )

        messages += data_point["output"]
        tokenized_full_prompt = self.tokenize(messages)

        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    def _load_dataset(self):
        if not self.using_raw_data:
            train_data = load_dataset('json', data_files="./data/train_data_qwen_100k.json")["train"]
            val_data = load_dataset('json', data_files="./data/val_data_qwen_100k.json")["train"]
        else:
            if self.data_path.endswith(".json") or self.data_path.endswith(".jsonl"):
                data = load_dataset("json", data_files=self.data_path)
            else:
                data = load_from_disk(self.data_path)


            if self.val_set_size > 0:
                train_val = data["train"].train_test_split(
                    test_size=self.val_set_size, shuffle=True, seed=42
                )

                train_data = train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
                val_data = train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
            else:
                train_data = data["train"].shuffle().map(self.generate_and_tokenize_prompt)
                val_data = None

            train_data.to_json("data/train_data_qwen_100k.json")
            if val_data:
                val_data.to_json("data/val_data_qwen_100k.json")

        return train_data, val_data