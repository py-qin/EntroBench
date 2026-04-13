### GSM8K
# import json
# import random
# from datasets import load_dataset
# import re


# def extract_final_answer(answer: str):
#     match = re.search(r"####\s*(.+)$", answer.strip())
#     if match:
#         return match.group(1).strip()
#     else:
#         return None


# random.seed(42)

# ds = load_dataset("openai/gsm8k", "main")
# test_ds = ds["test"]
# # randomly sample 500
# sampled_indices = random.sample(range(len(test_ds)), k=500)
# sampled_data = test_ds.select(sampled_indices)

# processed_examples = []
# for example in sampled_data:
#     q = example["question"]
#     final_answer = extract_final_answer(example["answer"])
#     prompt = f"The following is a math question:\nQuestion:{q}\nYou must first give the final answer.(e.g. Answer: 10)\nAnd then analyze with coherent sentence.".strip()
#     new_example = {
#         "question": prompt,
#         "answer": example["answer"],   
#         "final_answer": final_answer
#     }
#     processed_examples.append(new_example)

# output_path = "./dataset/GSM8K/GSM8K_prompt.jsonl"
# with open(output_path, "w", encoding="utf-8") as f:
#     for ex in processed_examples:
#         f.write(json.dumps(ex, ensure_ascii=False) + "\n")

#### MATH-500
# import json
# from datasets import load_dataset

# ds = load_dataset("HuggingFaceH4/MATH-500")
# test_ds = ds["test"]
# output_path = "./dataset/math500/math500_prompt.jsonl"

# with open(output_path, "w", encoding="utf-8") as f:
#     for example in test_ds:
#         q = example["problem"]

#         prompt = f"The following is a math question:\nQuestion:{q}\nYou must first give the final answer.(e.g. Answer: 10)\nAnd then analyze with coherent sentence.".strip()
#         record = {
#             "problem": prompt,
#             "solution": example["solution"],
#             "answer": example["answer"]
#         }
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")

### MMLU
# from datasets import load_dataset
# import json

# ds = load_dataset("cais/mmlu", "high_school_computer_science")
# test_ds = ds["test"]
# output_path = "./dataset/mmlu/mmlu.jsonl"

# with open(output_path, "w", encoding="utf-8") as f:
#     for example in test_ds:
#         record = {
#             "question": example["question"],
#             "choices": example["choices"],
#             "answer": example["answer"]
#         }
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")
### MMLU PROCESS
# from openai import OpenAI
# import json


# client = OpenAI(
#     api_key="",
#     base_url="",
# )

# with open("./dataset/mmlu/mmlu.jsonl", 'r', encoding='utf-8') as fin, \
#         open("./dataset/mmlu/mmlu_processed.jsonl", 'w', encoding='utf-8') as fout:
#     for line in fin:
#         line = line.strip()
#         if not line:
#             continue
#         data = json.loads(line)
        
#         question = data.get("question", "")
#         choices = data.get("choices", [])
        
#         prompt = f"The following is a multiple choice question about computer science:\nQuestion:{question}\nChoice:\n0:{choices[0]}\n1:{choices[1]}\n2:{choices[2]}\n3:{choices[3]}\nYou must first give the final answer:\nAnswer: 0, 1, 2 or 3.\n And then analyze with coherent sentence.".strip()
#         data["prompt"] = prompt

#         completion = client.chat.completions.create(
#             model="",
#             messages=[
#                 {'role': 'system', 'content': 'You are a helpful assistant.'},
#                 {'role': 'user', 'content': prompt}
#             ],
#             max_tokens=200
#         )
#         answer = completion.choices[0].message.content
#         data["response"] = answer

#         fout.write(json.dumps(data, ensure_ascii=False) + '\n')

### MT
# from datasets import load_dataset
# import json
# import random

# ds = load_dataset("haoranxu/WMT23-Test", "en-zh")

# test_ds = ds["test"]
# output_path = "./dataset/mt/mt_control_length.jsonl"

# en_sentences = []
# filtered_examples = []

# for example in test_ds:
#     en_text = example["en-zh"]["en"]
#     en_len = len(en_text)
#     en_sentences.append(en_len)
#     filtered_examples.append(example)

# avg_len = sum(en_sentences) / len(en_sentences)

# # print(avg_len) # 95

# long_examples = [
#     ex for ex in filtered_examples
#     if len(ex["en-zh"]["en"]) > 180 and len(ex["en-zh"]["en"]) < (240)
# ]

# sampled_examples = random.sample(long_examples, min(100, len(long_examples)))

# with open(output_path, "w", encoding="utf-8") as f:
#     for example in sampled_examples:
#         record = {
#             "en": example["en-zh"]["en"],
#             "zh": example["en-zh"]["zh"],
#         }
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")

## MT PROCESS

# import json


# with open("./dataset/mt/mt_control_length.jsonl", 'r', encoding='utf-8') as fin, \
#         open("./dataset/mt/mt_control_length_prompt.jsonl", 'w', encoding='utf-8') as fout:
#     for line in fin:
#         line = line.strip()
#         if not line:
#             continue
#         data = json.loads(line)
        
#         en = data.get("en", "")
#         zh = data.get("zh", "")
        
#         prompt = f"Translate the following English text ino Chinese.(Output only the translation and nothing else!)\nEnglish text: {en}".strip()
#         data["prompt"] = prompt
#         fout.write(json.dumps(data, ensure_ascii=False) + '\n')

### MBPP
from datasets import load_dataset
import json


# ds = load_dataset("google-research-datasets/mbpp", "sanitized")

# test_ds = ds["test"]
# output_path = "./dataset/mbpp/mbpp.jsonl"

# with open(output_path, "w", encoding="utf-8") as f:
#     for example in test_ds:
#         record = {
#             "prompt": example["prompt"],
#             "code": example["code"],
#             "test_list": example["test_list"]
#         }
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")