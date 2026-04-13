# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===========================================
# dataset.py
# Description: Dataset classes for evaluation
# ===========================================

import json
import re
from datasets import load_dataset


class BaseDataset:
    """Base class for dataset."""

    def __init__(self, max_samples: int = 200):
        """
        Initialize the dataset.
        
        Parameters:
            max_samples (int): Maximum number of samples to load. Default is 200.
        """
        self.max_samples = max_samples
        self.prompts = []
        self.natural_texts = []
        self.references = []

    @property
    def prompt_nums(self):
        """Return the number of prompts."""
        return len(self.prompts)

    @property
    def natural_text_nums(self):
        """Return the number of natural texts."""
        return len(self.natural_texts)

    @property
    def reference_nums(self):
        """Return the number of references."""
        return len(self.references)

    def get_prompt(self, index):
        """Return the prompt at the specified index."""
        return self.prompts[index]

    def get_natural_text(self, index):
        """Return the natural text at the specified index."""
        return self.natural_texts[index]

    def get_reference(self, index):
        """Return the reference at the specified index."""
        return self.references[index]

    def load_data(self):
        """Load and process data to populate prompts, natural_texts, and references."""
        pass


class MTDataset(BaseDataset):
    def __init__(self, data_source: str, max_samples: int = 100):
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        with open(self.data_source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['en'])

class MMLUDataset(BaseDataset):
    def __init__(self, data_source: str, max_samples: int = 100):
        self.final_answers = []
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        with open(self.data_source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['response'])
            self.final_answers.append(item['answer'])


class MixCodeDataset(BaseDataset):
    def __init__(self, data_source: str, max_samples: int = 200):
        self.final_answers = []
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        with open(self.data_source, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data[:self.max_samples]:
            q = item['query']
            prompt = "Provide a single solution. Python code first, then explanation and formulas.\n"
            prompt = prompt + "The problem is: " + q
            self.prompts.append(prompt)
            self.natural_texts.append(item['full_solution'])
            self.final_answers.append(item['code'])

class MATH500Dataset(BaseDataset):
    def __init__(self, data_source: str, max_samples: int = 500):
        self.final_answers = []
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        with open(self.data_source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['problem'])
            self.natural_texts.append(item['solution'])
            self.final_answers.append(item['answer']) # final answer

class GSM8KDataset(BaseDataset):
    def __init__(self, data_source: str, max_samples: int = 500):
        self.final_answers = []
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        with open(self.data_source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['question'])
            self.natural_texts.append(item['answer'])
            self.final_answers.append(item['final_answer']) # final answer


class C4Dataset(BaseDataset):
    """Dataset class for C4 dataset."""

    def __init__(self, data_source: str, max_samples: int = 200):
        """
            Initialize the C4 dataset.

            Parameters:
                data_source (str): The path to the C4 dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_source, 'r') as f:
           lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['natural_text'])


class WMT16DE_ENDataset(BaseDataset):
    """Dataset class for WMT16 DE-EN dataset."""

    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """
            Initialize the WMT16 DE-EN dataset.

            Parameters:
                data_source (str): The path to the WMT16 DE-EN dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the WMT16 DE-EN dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(item['de'])
            self.references.append(item['en'])

class CNN_DailyMailDataset(BaseDataset):
    """Dataset class for CNN/DailyMail dataset."""

    def __init__(self, data_source: str, max_samples: int = 200, global_prompt="Please summarize the following article: ") -> None:
        """
            Initialize the CNN/DailyMail dataset.

            Parameters:
                data_source (str): The path to the CNN/DailyMail dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.global_prompt = global_prompt
        self.load_data()
    
    def load_data(self):
        """Load data from the CNN/DailyMail dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(f"{self.global_prompt}{item['article']}")
            self.references.append(item['highlights'])


class HumanEvalDataset(BaseDataset):
    """Dataset class for HumanEval dataset."""

    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        """
            Initialize the HumanEval dataset.

            Parameters:
                data_source (str): The path to the HumanEval dataset file.
        """
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()
    
    def load_data(self):
        """Load data from the HumanEval dataset file."""
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            # process prompt
            prompt = item['prompt']
            sections = prompt.split(">>>")
            prompt = sections[0]
            if len(sections) > 1:
                prompt += '\"\"\"'

            self.prompts.append(prompt)
            self.references.append({'task': prompt, 'test': item['test'], 'entry_point': item['entry_point']})
            self.natural_texts.append(item['canonical_solution']) #add

class MBPPDataset(BaseDataset):
    def __init__(self, data_source: str, max_samples: int = 200) -> None:
        super().__init__(max_samples)
        self.data_source = data_source

        self.stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/"]
        self.dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
        self.load_data()

    def load_data(self):
        with open(self.data_source, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)

            prompt = "The following is a python example:\n"
            train_data = self.dataset['train'][0]
            prompt += f'"""\n{train_data["prompt"]} Your code should satisfy these tests:\n'
            prompt += '\n'.join(train_data['test_list']) + '\n"""\n'
            prompt += train_data['code'] + "\n"


            prompt += f'"""\n**Only return python function completion!**\nProblem:{item["prompt"]} Your code should satisfy these tests:\n'
            prompt += '\n'.join(item['test_list']) + '\n"""\n'

            self.prompts.append(prompt)
            # self.references.append({'task': prompt, 'test': item['test'], 'entry_point': item['entry_point']})
            self.natural_texts.append(item['code'])

    # @staticmethod
    def first_block(self, string):
        """Split off first block of code by scanning for class, def etc. on newlines."""
        return re.split("|".join(self.stop_words), string)[0].rstrip()



if __name__ == '__main__':
    # d1 = C4Dataset('/data2/qpy/MarkLLM-main/dataset/c4/processed_c4.json', max_samples=500)
    # d2 = WMT16DE_ENDataset('dataset/wmt16_de_en/validation.jsonl', max_samples=100)
    # d3 = HumanEvalDataset('dataset/HumanEval/test.jsonl', max_samples=100)
    # dataset = GSM8KDataset("/data2/qpy/MarkLLM-main/dataset/GSM8K/GSM8K.jsonl")
    # mt = MTDataset('/data2/qpy/MarkLLM-main/dataset/mt/mt_processed.jsonl', max_samples=40)
    # print(mt.get_prompt(0),end="")
    # print(mt.get_natural_text(0))
    # print(d1.prompt_nums)

    m = MBPPDataset("/data2/qpy/MarkLLM-main/dataset/mbpp/mbpp.jsonl",max_samples=40)
    print(m.get_prompt(0))
    print("\n\n===\n\n")
    print(m.get_natural_text(0))
