import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from evaluation.dataset import C4Dataset, GSM8KDataset, MixCodeDataset, MBPPDataset, MATH500Dataset, MMLUDataset, MTDataset


class SpikeEntropyCalculator:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-13b-chat-hf",
        z_value: float = 0.7615941589914151,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512
    ):
        self.z_value = z_value
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Llama tokenizer doesn't have pad_token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
        self.model.eval()

    def calculate_spike_entropy_for_sequence(self, text: str) -> float:
        """
        Calculate average spike entropy for a single sequence (excluding first token).
        Returns: average spike entropy over valid positions.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
            add_special_tokens=False
        ).to(self.device)

        input_ids = inputs["input_ids"]  # shape: [1, seq_len]
        seq_len = input_ids.shape[1]

        if seq_len <= 1:
            return 0.0

        with torch.no_grad():
            outputs = self.model(input_ids, return_dict=True)
            logits = outputs.logits

            logits = logits[:, :-1, :]  # [1, seq_len-1, vocab_size]
            probs = torch.softmax(logits, dim=-1)  # [1, seq_len-1, vocab_size]

            # Spike Entropy: sum_k p_k / (1 + z * p_k)
            denom = 1.0 + self.z_value * probs
            spike_vals = probs / denom  # [1, seq_len-1, vocab_size]
            spike_entropy_per_pos = spike_vals.sum(dim=-1)  # [1, seq_len-1]

            avg_spike_entropy = spike_entropy_per_pos.mean().item()
            return avg_spike_entropy

    def calculate_spike_entropy_for_natural_only(self, prompt: str, natural: str) -> float:
        """
        Calculate average spike entropy only over the 'natural' part.
        The model sees (prompt + natural) as input, but entropy is computed only for positions predicting 'natural' tokens.
        """
        # Tokenize separately to get lengths
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        natural_ids = self.tokenizer(natural, return_tensors="pt", add_special_tokens=False).input_ids[0]

        prompt_len = prompt_ids.shape[0]
        natural_len = natural_ids.shape[0]

        if natural_len == 0:
            return 0.0

        # Concatenate
        full_ids = torch.cat([prompt_ids, natural_ids], dim=0).unsqueeze(0).to(self.device)  # [1, total_len]

        with torch.no_grad():
            outputs = self.model(full_ids, return_dict=True)
            logits = outputs.logits  # [1, total_len - 1, vocab_size]

            # The logits that predict the 'natural' tokens start at index = prompt_len - 1
            # and go up to (prompt_len - 1 + natural_len - 1) = prompt_len + natural_len - 2
            start_idx = prompt_len - 1
            end_idx = start_idx + natural_len  # logits slice is [start:end], so end = start + natural_len

            if start_idx < 0 or end_idx > logits.shape[1]:
                # Edge case: prompt is empty?
                # If prompt is empty, start_idx = -1 → invalid. Handle by using all logits for natural.
                if prompt_len == 0:
                    start_idx = 0
                    end_idx = natural_len
                else:
                    return 0.0

            natural_logits = logits[:, start_idx:end_idx, :]  # [1, natural_len, vocab_size]

            probs = torch.softmax(natural_logits, dim=-1)
            denom = 1.0 + self.z_value * probs
            spike_vals = probs / denom
            spike_entropy_per_pos = spike_vals.sum(dim=-1)  # [1, natural_len]
            avg_spike_entropy = spike_entropy_per_pos.mean().item()

            return avg_spike_entropy


    def compute_dataset_average_entropy(self, dataset) -> float:
        entropies = []

        for i, (prompt, natural) in enumerate(zip(dataset.prompts, dataset.natural_texts)):
            try:
                ent = self.calculate_spike_entropy_for_natural_only(prompt, natural)
                entropies.append(ent)
                if (i + 1) % 500 == 0:
                    print(f"Processed {i+1}/{len(dataset.prompts)} samples...")
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        if not entropies:
            return 0.0, 0.
        
        entropies = np.array(entropies)
        mean = float(np.mean(entropies))
        std = float(np.std(entropies, ddof=0))
        return mean, std
    
    def compute_dataset_average_entropy_humaneval(self, dataset) -> float:
        entropies = []

        for i, (prompt, natural) in enumerate(zip(dataset.prompts, dataset.natural_texts)):
            try:
                ent = self.calculate_spike_entropy_for_natural_only(prompt + "\n", natural)
                entropies.append(ent)
                if (i + 1) % 500 == 0:
                    print(f"Processed {i+1}/{len(dataset.prompts)} samples...")
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        if not entropies:
            return 0.0, 0.0

        entropies = np.array(entropies)
        mean = float(np.mean(entropies))
        std = float(np.std(entropies, ddof=0))
        return mean, std

    def compute_on_llm_user(self, prompt, generation) -> float:
        try:
            ent = self.calculate_spike_entropy_for_natural_only(prompt, generation)
        except Exception as e:
            print(f"Error: {e}")

        return ent


if __name__ == "__main__":
    c4 = C4Dataset('./dataset/c4/processed_c4.json',max_samples=500)
    Gs = GSM8KDataset('./dataset/GSM8K/GSM8K_prompt.jsonl',max_samples=500)
    Math = MATH500Dataset('./dataset/math500/math500_prompt.jsonl',max_samples=500)
    Mix = MixCodeDataset('./dataset/Mix/mix_code.json',max_samples=200)
    mmlu = MMLUDataset('./dataset/mmlu/mmlu_processed.jsonl', max_samples=100)
    mt = MTDataset('./dataset/mt/mt_control_length_prompt.jsonl', max_samples=100)
    mbpp = MBPPDataset('./dataset/mbpp/mbpp.jsonl', max_samples=200)

    gamma = 0.5
    delta = 1.7
    alpha = torch.exp(torch.tensor(delta)).item()
    z_value = ((1-gamma)*(alpha-1))/(1-gamma+(alpha*gamma))
    print("z: ",z_value)

    calculator = SpikeEntropyCalculator(
        model_name="meta-llama/Llama-2-7b-chat-hf", # Qwen/Qwen2.5-7B-Instruct  Llama2-7b   meta-llama/Llama-2-13b-chat-hf
        z_value=z_value,
    )

    avg_entropy, std = calculator.compute_dataset_average_entropy(c4)
    print(f"\nC4 average Spike Entropy: {avg_entropy:.4f} std:{std:.4f}")
    avg_entropy, std  = calculator.compute_dataset_average_entropy(mt)
    print(f"\nMT average Spike Entropy: {avg_entropy:.4f} std:{std:.4f}")

    avg_entropy, std = calculator.compute_dataset_average_entropy(mmlu)
    print(f"\nMMLU average Spike Entropy: {avg_entropy:.4f} std:{std:.4f}")

    avg_entropy, std = calculator.compute_dataset_average_entropy(Math)
    print(f"\nMath500 average Spike Entropy: {avg_entropy:.4f} std:{std:.4f}")

    avg_entropy, std = calculator.compute_dataset_average_entropy(Gs)
    print(f"\nGSM8K average Spike Entropy: {avg_entropy:.4f} std:{std:.4f}")

    avg_entropy, std = calculator.compute_dataset_average_entropy(Mix)
    print(f"\nMixCode average Spike Entropy: {avg_entropy:.4f} std:{std:.4f}")

    avg_entropy, std = calculator.compute_dataset_average_entropy_humaneval(mbpp)
    print(f"\nMBPP average Spike Entropy: {avg_entropy:.4f} std:{std:.4f}")
