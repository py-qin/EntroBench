import torch
import argparse
import json
import re
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.dataset import (C4Dataset, MTDataset,
                                WMT16DE_ENDataset, HumanEvalDataset, CNN_DailyMailDataset, 
                                GSM8KDataset, MATH500Dataset, MMLUDataset,
                                MBPPDataset,
                                MixCodeDataset)
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, LlamaTokenizer
from evaluation.tools.text_editor import TruncatePromptTextEditor, PromptTextEditor, SummarizeTextEditor, InfoTextEditor, CodeOnlyTextEditor
from evaluation.tools.text_quality_analyzer import (PPLCalculator, BLEUCalculator, PSPCalculator,
                                                    PassOrNotJudger, GPTTextDiscriminator, ROUGE1Calculator, 
                                                    ROUGE2Calculator, ROUGELCalculator, BERTScoreCalculator)
from evaluation.pipelines.detection import WatermarkedTextDetectionPipeline, UnWatermarkedTextDetectionPipeline, DetectionPipelineReturnType
from evaluation.pipelines.quality_analysis import (DirectTextQualityAnalysisPipeline, ReferencedTextQualityAnalysisPipeline, ExternalDiscriminatorTextQualityAnalysisPipeline, 
                                                   QualityPipelineReturnType,
                                                   PreloadedDirectTextQualityPipeline)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def code_detection_pipeline(model_name, dataset_name, watermark_name, output_dir):
    if dataset_name == "MIX":
        my_dataset = MixCodeDataset('./dataset/Mix/mix_code.json',max_samples=200)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=200,
                                            do_sample=True,
                                            # min_length=200,
                                            min_new_tokens=150,
                                            no_repeat_ngram_size=4)
    
    elif dataset_name == "Human":
        my_dataset = HumanEvalDataset('./dataset/human_eval/test.jsonl', max_samples=1)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                                tokenizer=tokenizer,
                                                vocab_size=model.config.vocab_size,
                                                device=device,
                                                min_length=200,
                                                max_length=300)

    elif dataset_name == "mbpp":
        my_dataset = MBPPDataset('./dataset/mbpp/mbpp.jsonl', max_samples=200)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                                tokenizer=tokenizer,
                                                vocab_size=model.config.vocab_size,
                                                device=device,
                                                # min_new_tokens=15
                                                max_new_tokens=200,
                                                min_new_tokens=150
                                                )
        
    else:
        print("Invalid dataset")
        return


    my_watermark = AutoWatermark.load(watermark_name, 
                                    algorithm_config=f'config/{watermark_name}.json',
                                    transformers_config=transformers_config)

    pipeline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[PromptTextEditor()],
                                                show_progress=True,
                                                return_type=DetectionPipelineReturnType.SCORES,
                                                output_dir=output_dir) 

    pipeline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True,
                                                return_type=DetectionPipelineReturnType.SCORES,
                                                output_dir=output_dir)

    result1 = pipeline1.evaluate(my_watermark)
    result2 = pipeline2.evaluate(my_watermark)

    # save
    model_name = model_name.replace("/", "--")
    pipeline1.save_results(model_name, dataset_name, watermark_name)
    pipeline2.save_results(model_name, dataset_name, watermark_name)

    calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='target_fpr', target_fpr=0.01)
    metrics = calculator.calculate(result1, result2)
    print(metrics)

def test_detection_pipeline(model_name, dataset_name, watermark_name, output_dir):
    if dataset_name == "c4":
        my_dataset = C4Dataset('./dataset/c4/processed_c4.json',max_samples=100)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # print("tokenizer len:",len(tokenizer))
        # print("Model vocab_size from config:", model.config.vocab_size)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=230,
                                            do_sample=True,
                                            # min_length=230,
                                            min_new_tokens=200,
                                            no_repeat_ngram_size=4)

    elif dataset_name == "GSM8K":
        my_dataset = GSM8KDataset('./dataset/GSM8K/GSM8K_prompt.jsonl',max_samples=200)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=200,
                                            do_sample=True,
                                            # min_length=200,
                                            # min_new_tokens=50,
                                            no_repeat_ngram_size=4)

    elif dataset_name == "MATH500":
        my_dataset = MATH500Dataset('./dataset/math500/math500_prompt.jsonl',max_samples=200)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=200,
                                            do_sample=True,
                                            # min_length=200,
                                            # min_new_tokens=200,
                                            no_repeat_ngram_size=4)

    elif dataset_name == "MIX":
        my_dataset = MixCodeDataset('./dataset/Mix/mix_code.json',max_samples=200)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=230,
                                            do_sample=True,
                                            # min_length=200,
                                            min_new_tokens=200,
                                            no_repeat_ngram_size=4)

    elif dataset_name == "WMT":
        # my_dataset = MTDataset('./dataset/mt/mt_processed.jsonl',max_samples=100)
        my_dataset = MTDataset('./dataset/mt/mt_control_length_prompt.jsonl',max_samples=100)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=100,
                                            do_sample=True,
                                            # min_length=200,
                                            no_repeat_ngram_size=4)

    elif dataset_name == "MMLU":
        my_dataset = MMLUDataset('./dataset/mmlu/mmlu_processed.jsonl',max_samples=100)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=200,
                                            do_sample=True,
                                            # min_length=200,
                                            # min_new_tokens=200,
                                            no_repeat_ngram_size=4)
        
    else:
        print("Invalid dataset")
        return

    my_watermark = AutoWatermark.load(watermark_name, 
                                    algorithm_config=f'config/{watermark_name}.json',
                                    transformers_config=transformers_config)

    pipeline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[TruncatePromptTextEditor()],  #TruncatePromptTextEditor delete dataset's prompt
                                                show_progress=True,
                                                return_type=DetectionPipelineReturnType.SCORES,
                                                output_dir=output_dir) 

    pipeline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset, text_editor_list=[],
                                                show_progress=True,
                                                return_type=DetectionPipelineReturnType.SCORES,
                                                output_dir=output_dir)

    result1 = pipeline1.evaluate(my_watermark)
    result2 = pipeline2.evaluate(my_watermark)

    # save
    model_name = model_name.replace("/", "--")
    pipeline1.save_results(model_name, dataset_name, watermark_name)
    pipeline2.save_results(model_name, dataset_name, watermark_name)

    calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='target_fpr', target_fpr=0.01)
    metrics = calculator.calculate(result1, result2)
    print(metrics)


def load_texts_from_json(file_path, key):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {file_path}")
    return [item[key] for item in data]

# PPL
def test_direct_quality_analysis_pipeline_gen(model_name, dataset_name, watermark_name, metric):
    if metric == "Bert":
        my_dataset = C4Dataset('./dataset/c4/processed_c4.json',max_samples=200)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # print("tokenizer len:",len(tokenizer))
        # print("Model vocab_size from config:", model.config.vocab_size)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=230,
                                            do_sample=True,
                                            # min_length=230,
                                            min_new_tokens=200,
                                            no_repeat_ngram_size=4)
        my_watermark = AutoWatermark.load(watermark_name,
                                        algorithm_config=f'config/{watermark_name}.json',
                                        transformers_config=transformers_config)
        
        quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset,
                                                                watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                                unwatermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                                analyzers=[BERTScoreCalculator(model_path="microsoft/deberta-xlarge-mnli")],
                                                                unwatermarked_text_source='generated', show_progress=True,
                                                                return_type=QualityPipelineReturnType.MEAN_SCORES)
        print(quality_pipeline.evaluate(my_watermark))

    elif metric == "PSP":
        my_dataset = C4Dataset('./dataset/c4/processed_c4.json',max_samples=200)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # print("tokenizer len:",len(tokenizer))
        # print("Model vocab_size from config:", model.config.vocab_size)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=230,
                                            do_sample=True,
                                            # min_length=230,
                                            min_new_tokens=200,
                                            no_repeat_ngram_size=4)
        my_watermark = AutoWatermark.load(watermark_name,
                                        algorithm_config=f'config/{watermark_name}.json',
                                        transformers_config=transformers_config)
        
        quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset,
                                                                watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                                unwatermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                                analyzers=[PSPCalculator()],
                                                                unwatermarked_text_source='generated', show_progress=True,
                                                                return_type=QualityPipelineReturnType.MEAN_SCORES)
        print(quality_pipeline.evaluate(my_watermark))
    
    elif metric == "PPL":
        my_dataset = C4Dataset('./dataset/c4/processed_c4.json',max_samples=200)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                                    tokenizer=tokenizer,
                                                    vocab_size=model.config.vocab_size,
                                                    device=device,
                                                    max_new_tokens=230,
                                                    do_sample=True,
                                                    # min_length=230,
                                                    min_new_tokens=200,
                                                    no_repeat_ngram_size=4)
        my_watermark = AutoWatermark.load(watermark_name,
                                        algorithm_config=f'config/{watermark_name}.json',
                                        transformers_config=transformers_config)
        quality_pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                            watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                            unwatermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                            analyzers=[PPLCalculator(model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map='auto'),
                                                                                    tokenizer=LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf"),
                                                                                    device=device)],
                                                            unwatermarked_text_source='generated', show_progress=True, 
                                                            return_type=QualityPipelineReturnType.MEAN_SCORES)
        print(quality_pipeline.evaluate(my_watermark))

    elif metric == "pass_GSM8K":
        my_dataset = GSM8KDataset('./dataset/GSM8K/GSM8K_prompt.jsonl',max_samples=200)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                                    tokenizer=tokenizer,
                                                    vocab_size=model.config.vocab_size,
                                                    device=device,
                                                    max_new_tokens=200,
                                                    do_sample=True,
                                                    # min_length=230,
                                                    # min_new_tokens=200,
                                                    no_repeat_ngram_size=4)
        my_watermark = AutoWatermark.load(watermark_name,
                                        algorithm_config=f'config/{watermark_name}.json',
                                        transformers_config=transformers_config)
        quality_pipeline = DirectTextQualityAnalysisPipeline(dataset=my_dataset,
                                                            watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                            unwatermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                            analyzers=[PPLCalculator(model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map='auto'),
                                                                                    tokenizer=LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf"),
                                                                                    device=device)],
                                                            unwatermarked_text_source='generated', show_progress=True, 
                                                            return_type=QualityPipelineReturnType.MEAN_SCORES)
        print(quality_pipeline.evaluate(my_watermark))
    
    elif metric == "BLEU":
        my_dataset = MTDataset('./dataset/mt/mt_control_length_prompt.jsonl',max_samples=100)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=100,
                                            do_sample=True,
                                            # min_length=200,
                                            no_repeat_ngram_size=4)
        
        my_watermark = AutoWatermark.load(watermark_name,
                                        algorithm_config=f'config/{watermark_name}.json',
                                        transformers_config=transformers_config)

        quality_pipeline = ReferencedTextQualityAnalysisPipeline(dataset=my_dataset, 
                                                                watermarked_text_editor_list=[TruncatePromptTextEditor()],
                                                                unwatermarked_text_editor_list=[],
                                                                analyzers=[BLEUCalculator()],
                                                                unwatermarked_text_source='natural', show_progress=True, 
                                                                return_type=QualityPipelineReturnType.MEAN_SCORES)
        print(quality_pipeline.evaluate(my_watermark))

def test_direct_quality_analysis_pipeline_1(model_name, dataset_name, watermark_name, metric):
    model_name = model_name.replace("/", "--")
    w_json = f"./outputs/{model_name}/{dataset_name}/{watermark_name}_watermarked.json"
    unw_json = f"./outputs/{model_name}/{dataset_name}/{watermark_name}_unwatermarked.json"

    if metric == "PPL":
        analyzer = PPLCalculator(model=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map='auto'),
                                                                                    tokenizer=LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf"),
                                                                                    device=device)

        pipeline = PreloadedDirectTextQualityPipeline(
                watermarked_text_editor_list=[PromptTextEditor()],
                analyzers=[analyzer],
                show_progress=True,
                return_type=QualityPipelineReturnType.MEAN_SCORES
                )

        result = pipeline.evaluate_from_files(w_json, unw_json)
        print(result)
    
    elif metric == "Pass":
        with open(w_json, 'r', encoding='utf-8') as f:
            wm_data = json.load(f)
            if not isinstance(wm_data, list):
                raise ValueError("Watermarked JSON must be a list of objects.")
            watermarked_texts = [item['generated_text'] for item in wm_data]
            prompt_list = [item['prompt'] for item in wm_data]

        with open(unw_json, 'r', encoding='utf-8') as f:
            uwm_data = json.load(f)
            if not isinstance(uwm_data, list):
                raise ValueError("Unwatermarked JSON must be a list of objects.")
            unwatermarked_texts = [item['text'] for item in uwm_data]

        editor = PromptTextEditor()
        total = 0
        pred_t = 0
        if dataset_name == "GSM8K":
            for i in range(len(watermarked_texts)):
                wm = editor.edit(watermarked_texts[i],prompt_list[i])
                ## match
                match = re.search(r'Answer.*?(\d+)', wm)
                pred = int(match.group(1)) if match else 0

                match = re.search(r'####.*?(\d+)', unwatermarked_texts[i])
                true_label = int(match.group(1)) if match else 0

                # print("pred:",pred, "true:",true_label)

                if pred == true_label:
                    pred_t +=1
                total += 1
      
            print("Pass@1:",pred_t / total)
        
        elif dataset_name == "MATH500":
            my_dataset = MATH500Dataset('./dataset/math500/math500_prompt.jsonl',max_samples=200)
            final_answer = my_dataset.final_answers
            from openai import OpenAI
            import time
            from tqdm import tqdm
            client = OpenAI(
                api_key="",
                base_url="",
            )
            for i in tqdm(range(len(watermarked_texts)), desc="Judging"):
                wm = editor.edit(watermarked_texts[i],prompt_list[i])
                
                time.sleep(1)

                prompt = "Return True if the given answer matches the correct answer mathematically; otherwise, return False. Only output True or False.\n"
                prompt += f"Given answer: {wm}\n. Correct answer: {final_answer[i]}. Output: "
                completion = client.chat.completions.create(
                    model="",
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    max_tokens=10
                )
                answer = completion.choices[0].message.content

                if answer == "True":
                    pred_t +=1
                total += 1

            print("Pass@1:",pred_t / total)

    elif metric == "Pass_code":
        with open(w_json, 'r', encoding='utf-8') as f:
            wm_data = json.load(f)
            if not isinstance(wm_data, list):
                raise ValueError("Watermarked JSON must be a list of objects.")
            watermarked_texts = [item['generated_text'] for item in wm_data]
            prompt_list = [item['prompt'] for item in wm_data]

        with open(unw_json, 'r', encoding='utf-8') as f:
            uwm_data = json.load(f)
            if not isinstance(uwm_data, list):
                raise ValueError("Unwatermarked JSON must be a list of objects.")
            unwatermarked_texts = [item['text'] for item in uwm_data]

        editor = PromptTextEditor()
        total = 0
        pred_t = 0

        from openai import OpenAI
        import time
        from tqdm import tqdm
        client = OpenAI(
            api_key="",
            base_url="",
        )
        for i in tqdm(range(len(watermarked_texts)), desc="Judging"):
            wm = editor.edit(watermarked_texts[i],prompt_list[i])
            # time.sleep(1)
            prompt = "Return True if the given code correctly solves the problem; otherwise, return False. Only output True or Flase.\n"
            prompt += f"Given code: {wm}\n. Problem: {prompt_list[i]}. Output: "
            completion = client.chat.completions.create(
                model="",
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=10
            )
            answer = completion.choices[0].message.content

            if answer == "True":
                pred_t +=1
            total += 1

        print("Pass@1:",pred_t / total)

    elif metric == "ACC":
        with open(w_json, 'r', encoding='utf-8') as f:
            wm_data = json.load(f)
            if not isinstance(wm_data, list):
                raise ValueError("Watermarked JSON must be a list of objects.")
            watermarked_texts = [item['generated_text'] for item in wm_data]
            prompt_list = [item['prompt'] for item in wm_data]

        with open(unw_json, 'r', encoding='utf-8') as f:
            uwm_data = json.load(f)
            if not isinstance(uwm_data, list):
                raise ValueError("Unwatermarked JSON must be a list of objects.")
            unwatermarked_texts = [item['text'] for item in uwm_data]

        editor = PromptTextEditor()
        my_dataset = MMLUDataset('./dataset/mmlu/mmlu_processed.jsonl',max_samples=100)
        final_answer = my_dataset.final_answers
        total = 0
        pred_t = 0
        for i in range(len(watermarked_texts)):
            wm = editor.edit(watermarked_texts[i],prompt_list[i])

            match = re.search(r'Answer.*?(\d+)', wm)
            pred = int(match.group(1)) if match else 0

            # print("pred:",pred, "true:",true_label)

            if pred == final_answer[i]:
                pred_t +=1
            total += 1
    
        print("ACC:",pred_t / total)

def llm_as_user_pipeline(model_name, dataset_name, watermark_name, output_dir):
    if dataset_name == "c4":
        my_dataset = C4Dataset('./dataset/c4/processed_c4.json',max_samples=100)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=230,
                                            do_sample=True,
                                            # min_length=230,
                                            min_new_tokens=200,
                                            no_repeat_ngram_size=4)

    elif dataset_name == "MIX":
        my_dataset = MixCodeDataset('./dataset/Mix/mix_code.json',max_samples=200)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        transformers_config = TransformersConfig(model=model,
                                            tokenizer=tokenizer,
                                            vocab_size=model.config.vocab_size,
                                            device=device,
                                            max_new_tokens=230,
                                            do_sample=True,
                                            # min_length=200,
                                            min_new_tokens=200,
                                            no_repeat_ngram_size=4)
        
    else:
        print("Invalid dataset")
        return

    my_watermark = AutoWatermark.load(watermark_name, 
                                    algorithm_config=f'config/{watermark_name}.json',
                                    transformers_config=transformers_config)

    pipeline1 = WatermarkedTextDetectionPipeline(dataset=my_dataset, 
                                                 text_editor_list=[TruncatePromptTextEditor(), InfoTextEditor()], # SummarizeTextEditor  | CodeOnlyTextEditor
                                                show_progress=True,
                                                return_type=DetectionPipelineReturnType.SCORES,
                                                output_dir=output_dir) 

    pipeline2 = UnWatermarkedTextDetectionPipeline(dataset=my_dataset,
                                                   text_editor_list=[InfoTextEditor()], # SummarizeTextEditor  | CodeOnlyTextEditor
                                                show_progress=True,
                                                return_type=DetectionPipelineReturnType.SCORES,
                                                output_dir=output_dir)

    result1 = pipeline1.evaluate(my_watermark)
    result2 = pipeline2.evaluate(my_watermark)

    calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='target_fpr', target_fpr=0.01)
    metrics = calculator.calculate(result1, result2)
    print(metrics, end="")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EntroBench")
    parser.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct") # meta-llama/Llama-2-7b-chat-hf
    parser.add_argument("-d", "--dataset", default="c4")  # c4 | WMT | GSM8K | MATH500 | MMLU | mbpp | MIX
    parser.add_argument("-w", "--watermark", default="KGW")

    args = parser.parse_args()

    # test_detection_pipeline(model_name=args.model,dataset_name=args.dataset,watermark_name=args.watermark, output_dir="")
    
    # code_detection_pipeline(model_name=args.model,dataset_name=args.dataset,watermark_name=args.watermark, output_dir="")
    
    # Quality
    # test_direct_quality_analysis_pipeline_1(model_name=args.model,dataset_name=args.dataset,watermark_name=args.watermark, metric="Pass")
    # test_direct_quality_analysis_pipeline_gen(model_name=args.model,dataset_name=args.dataset,watermark_name=args.watermark, metric="BLEU")

    # LLM AS A USER
    llm_as_user_pipeline(model_name=args.model,dataset_name=args.dataset,watermark_name=args.watermark,output_dir="")
