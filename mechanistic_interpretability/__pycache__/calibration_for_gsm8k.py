import torch
import pandas as pd
import re
import os
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import your recorder
# Ensure activation_recorder.py is in the same directory
try:
    from neuron_activation_recorder import NeuronActivationRecorder
except ImportError:
    raise ImportError("Please ensure 'activation_recorder.py' is in the same directory.")

class GSM8KEvaluator:
    def __init__(self, model, tokenizer, neuron_indices_path: str, layer_name: str, use_input: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.neuron_indices_path = neuron_indices_path
        self.layer_name = layer_name
        self.use_input = use_input
        
        # Load neuron indices to know column names later
        import json
        with open(neuron_indices_path, 'r') as f:
            self.neuron_indices = json.load(f)

    def extract_answer_gsm8k(self, text: str) -> str:
        """
        Extracts the answer from GSM8K examples or model output.
        GSM8K training data often uses '#### 42' to denote the answer.
        Model output might vary, so we look for the last number if no delimiter is found.
        """
        # 1. Try to find the GSM8K specific delimiter (often used in few-shot prompts)
        if '####' in text:
            return text.split('####')[-1].strip()
        
        # 2. Try to find \boxed{} (common in math fine-tunes)
        match = re.search(r'\\boxed\{([^{}]+)\}', text)
        if match:
            return match.group(1).strip()
            
        # 3. Fallback: Heuristic to find the last number in the text
        # Clean text to remove potential currency symbols or commas for parsing
        clean_text = text.replace(',', '').replace('$', '')
        numbers = re.findall(r'-?\d+\.?\d*', clean_text)
        if numbers:
            return numbers[-1]
            
        return ""

    def check_correctness(self, generated_answer: str, ground_truth: str) -> bool:
        """
        Comparing numbers can be tricky (e.g., 42.0 vs 42).
        We attempt to convert to float for comparison.
        """
        try:
            # Simple string match first
            if generated_answer.strip() == ground_truth.strip():
                return True
                
            # Float comparison
            gen_float = float(generated_answer)
            truth_float = float(ground_truth)
            return abs(gen_float - truth_float) < 1e-6
        except ValueError:
            return False

    def evaluate(self, num_samples: int = None, output_file: str = "gsm8k_analysis_results.csv"):
        """
        Runs the evaluation loop on GSM8K.
        """
        print("Loading GSM8K dataset...")
        # 'main' is the default config for GSM8K
        dataset = load_dataset("gsm8k", "main", split="test")

        # Limit samples
        if num_samples is not None:
            dataset = list(dataset)[:num_samples]
        
        all_records = []

        print(f"Starting evaluation on {len(dataset)} samples...")

        # Initialize Recorder
        recorder = NeuronActivationRecorder(
            self.model, 
            self.neuron_indices_path, 
            self.layer_name, 
            self.use_input
        )

        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
            question = sample['question']
            # GSM8K answers in the dataset look like "reasoning steps... #### answer"
            ground_truth = self.extract_answer_gsm8k(sample['answer'])
            
            # 1. Format Prompt
            # Standard few-shot style or simple instruction
            prompt = f"Question: {question}\nLet's think step by step.\nAnswer:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # 2. Run Inference with Recorder
            with recorder:
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=256,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 3. Process Text
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Isolate response from prompt
            response_only = generated_text[len(prompt):]
            
            extracted_ans = self.extract_answer_gsm8k(response_only)
            is_correct = self.check_correctness(extracted_ans, ground_truth)
            
            # 4. Process Activations
            activations = recorder.get_activations()
            
            if activations is not None:
                act_list = activations.cpu().tolist()
                
                # Check lengths to try to align tokens if possible (approximate)
                # Note: activations include prompt tokens. 
                
                for token_idx, act_values in enumerate(act_list):
                    row = {
                        "problem_id": i,
                        "token_index": token_idx,
                        "is_correct": is_correct,
                        "generated_text": response_only,
                        "ground_truth": ground_truth,
                        "extracted_answer": extracted_ans
                    }
                    # Add individual neuron columns
                    for n_idx, val in zip(self.neuron_indices, act_values):
                        row[f"Neuron_{n_idx}"] = val
                    
                    all_records.append(row)

        # 5. Save to Disk
        print(f"Saving {len(all_records)} rows of data to {output_file}...")
        df = pd.DataFrame(all_records)
        df.to_csv(output_file, index=False)
        print("Done.")

# --- Usage Example ---
if __name__ == "__main__":
    # Mock setup for testing the script structure
    # 1. Setup Mock Model (Replace with your Unsloth/Llama loading)
    # from unsloth import FastLanguageModel
    # model, tokenizer = FastLanguageModel.from_pretrained(...)
    
    # MOCK objects for running this script standalone
    try:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f"Device is set to: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B",
            torch_dtype=torch.bfloat16,
            device_map='mps', 
            low_cpu_mem_usage=True,
            trust_remote_code=True 
        )
         # Set the tokenizer for the model
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")
        print("Loaded in model successfully")
    except Exception as e:
        print(f"Failed to load model unsloth/Meta-Llama-3.1-8B: {e}")

    evaluator = GSM8KEvaluator(
        model=model,
        tokenizer=tokenizer,
        neuron_indices_path='data.json',
        layer_name='model.layers.31.mlp.down_proj', 
        use_input=True
    )
    
    # Run a quick test
    try:
        evaluator.evaluate(num_samples=30)
    except Exception as e:
        print(f"Test run skipped or failed (expected if datasets not installed): {e}")