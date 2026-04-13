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

# =============================================
# detection.py
# Description: Pipeline for watermark detection
# =============================================

from tqdm import tqdm
from enum import Enum, auto
from pathlib import Path
import json
import time
from dataset_level_entropy import SpikeEntropyCalculator
from watermark.base import BaseWatermark
from evaluation.dataset import BaseDataset
from evaluation.tools.text_editor import TextEditor
from exceptions.exceptions import InvalidTextSourceModeError


class DetectionPipelineReturnType(Enum):
    """Return type of the watermark detection pipeline."""
    FULL = auto()
    SCORES = auto()
    IS_WATERMARKED = auto()


class WatermarkDetectionResult:
    """Result of watermark detection."""

    def __init__(self, generated_or_retrieved_text, edited_text, detect_result) -> None:
        """
            Initialize the watermark detection result.

            Parameters:
                generated_or_retrieved_text: The generated or retrieved text.
                edited_text: The edited text.
                detect_result: The detection result.
        """
        self.generated_or_retrieved_text = generated_or_retrieved_text
        self.edited_text = edited_text
        self.detect_result = detect_result
        pass


class WatermarkDetectionPipeline:
    """Pipeline for watermark detection."""

    def __init__(self, dataset: BaseDataset, text_editor_list: list[TextEditor] = [], 
                 show_progress: bool = True, return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES) -> None:
        """
            Initialize the watermark detection pipeline.

            Parameters:
                dataset (BaseDataset): The dataset for the pipeline.
                text_editor_list (list[TextEditor]): The list of text editors.
                show_progress (bool): Whether to show progress bar.
                return_type (DetectionPipelineReturnType): The return type of the pipeline.
        """
        self.dataset = dataset
        self.text_editor_list = text_editor_list
        self.show_progress = show_progress
        self.return_type = return_type
        # self.EntroCal = SpikeEntropyCalculator()
       
    def _edit_text(self, text: str, prompt: str = None):
        """Edit text using text editors."""
        for text_editor in self.text_editor_list:
            text = text_editor.edit(text, prompt)
        return text
    
    def _generate_or_retrieve_text(self, dataset_index: int, watermark: BaseWatermark):
        """Generate or retrieve text from dataset."""
        pass

    def _detect_watermark(self, text: str, watermark: BaseWatermark):
        """Detect watermark in text."""
        detect_result = watermark.detect_watermark(text, return_dict=True)
        return detect_result

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        pass

    def _get_progress_bar(self, iterable):
        """Return an iterable possibly wrapped with a progress bar."""
        if self.show_progress:
            return tqdm(iterable, desc="Processing", leave=True)
        return iterable

    def evaluate(self, watermark: BaseWatermark):
        """Conduct evaluation utilizing the pipeline."""
        evaluation_result = []
        bar = self._get_progress_bar(self._get_iterable())
        err_n = 0
        all_entro = []

        for index in bar:
            generated_or_retrieved_text = self._generate_or_retrieve_text(index, watermark)
            # print("\n\n生成: ", generated_or_retrieved_text)
            try:
                edited_text = self._edit_text(generated_or_retrieved_text, self.dataset.get_prompt(index))
            except:
                err_n +=1
                edited_text = generated_or_retrieved_text
            # edited_text = self.dataset.first_block(edited_text)  #######!!!!
            # print("\n\n编辑: ", edited_text)
            try:
                # all_entro.append(self.EntroCal.compute_on_llm_user(self.dataset.get_prompt(index), edited_text))
                detect_result = self._detect_watermark(edited_text, watermark)
            except:
                # detect_result = None
                err_n +=1
                continue
            evaluation_result.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text, detect_result))
        
        if err_n > 0:
            print("skip:",err_n)
        # print("Avg Entro:",sum(all_entro)/len(all_entro))

        if self.return_type == DetectionPipelineReturnType.FULL:
            return evaluation_result
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [result.detect_result['score'] for result in evaluation_result]
        elif self.return_type == DetectionPipelineReturnType.IS_WATERMARKED:
            return [result.detect_result['is_watermarked'] for result in evaluation_result]


class WatermarkedTextDetectionPipeline(WatermarkDetectionPipeline):
    """Pipeline for detecting watermarked text."""

    def __init__(self, dataset, text_editor_list=[],
                 show_progress=True, return_type=DetectionPipelineReturnType.SCORES,
                 output_dir=None, *args, **kwargs) -> None:
        super().__init__(dataset, text_editor_list, show_progress, return_type)
        self.output_dir = output_dir
        self.generated_texts = []  # temp

    def _get_iterable(self):
        """Return an iterable for the prompts."""
        return range(self.dataset.prompt_nums)
    
    def _generate_or_retrieve_text(self, dataset_index, watermark):
        """Generate watermarked text from the dataset."""
        prompt = self.dataset.get_prompt(dataset_index)
        # print("\n\nPrompt: ",prompt)
        generated_text = watermark.generate_watermarked_text(prompt)
        self.generated_texts.append({
            "index": dataset_index,
            "prompt": prompt,
            "generated_text": generated_text
        })  # save to list

        return generated_text

    def save_results(self, model_name, dataset_name, watermark_name):
        if self.output_dir is None:
            return
        path = Path(self.output_dir) / model_name / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{watermark_name}_watermarked.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.generated_texts, f, ensure_ascii=False, indent=2)
        print(f"Saved watermarked texts to {file_path}")


class UnWatermarkedTextDetectionPipeline(WatermarkDetectionPipeline):
    """Pipeline for detecting unwatermarked text."""

    def __init__(self, dataset, text_editor_list=[], text_source_mode='natural',
                 show_progress=True, return_type=DetectionPipelineReturnType.SCORES,
                  output_dir=None,
                   *args, **kwargs) -> None:
        # Validate text_source_mode
        if text_source_mode not in ['natural', 'generated']:
            raise InvalidTextSourceModeError(text_source_mode)
        
        super().__init__(dataset, text_editor_list, show_progress, return_type)
        self.text_source_mode = text_source_mode
        self.output_dir = output_dir
        self.generated_texts = []  #tmp

    def _get_iterable(self):
        """Return an iterable for the natural texts or prompts."""
        if self.text_source_mode == 'natural':
            return range(self.dataset.natural_text_nums)
        else:
            return range(self.dataset.prompt_nums)
    
    def _generate_or_retrieve_text(self, dataset_index, watermark):
        """Retrieve unwatermarked text from the dataset."""
        if self.text_source_mode == 'natural':
            text = self.dataset.get_natural_text(dataset_index)
            prompt = self.dataset.get_prompt(dataset_index) if hasattr(self.dataset, 'get_prompt') else ""
        else:
            prompt = self.dataset.get_prompt(dataset_index)
            text = watermark.generate_unwatermarked_text(prompt)
        
        self.generated_texts.append({
            "index": dataset_index,
            "prompt": prompt,
            "text": text,
        })
        time.sleep(1)
        return text

    def save_results(self, model_name, dataset_name, watermark_name):
        if self.output_dir is None:
            return
        path = Path(self.output_dir) / model_name / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{watermark_name}_unwatermarked.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.generated_texts, f, ensure_ascii=False, indent=2)
        print(f"Saved unwatermarked texts to {file_path}")