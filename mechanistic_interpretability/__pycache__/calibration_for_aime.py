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

class AIMEEvaluator:
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

    def extract_answer(self, text: str) -> str:
        """
        Extracts the answer from the model output. 
        AIME usually expects an integer, often inside \boxed{}.
        """
        # Look for \boxed{answer}
        match = re.search(r'\\boxed\{([^{}]+)\}', text)
        if match:
            return match.group(1).strip()
        
        # Fallback: Look for the last number (simple heuristic)
        numbers = re.findall(r'-?\d+', text)
        if numbers:
            return numbers[-1]
        return ""

    def check_correctness(self, generated_answer: str, ground_truth: str) -> bool:
        """
        Simple comparison. Normalize strings to handle spacing/formatting.
        """
        clean_gen = generated_answer.strip().lower()
        clean_truth = ground_truth.strip().lower()
        return clean_gen == clean_truth

    def evaluate(self, num_samples: int = None, output_file: str = "aime_analysis_results.csv"):
        """
        Runs the evaluation loop.
        """
        # Load AIME dataset (using a common HF path, changeable)
        # 'lighteval/MATH' contains AIME subset, or 'HuggingFaceH4/aime_2024'
        print("Loading dataset...")
        try:
            # Fallback to a generic structure if specific subset fails
            dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        except:
            print("Warning: Could not load specific AIME dataset. Using a mock list for demonstration.")
            os.kill()

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
            problem = sample.get('problem', '')
            ground_truth = self.extract_answer(sample.get('solution', sample.get('answer', '')))
            
            # 1. Format Prompt (Modify based on your model's chat template)
            # Simple format for base models or instruction tuned
            prompt = f"Problem: {problem}\nSolve this step by step and put the final answer in \\boxed{{}}.\nSolution:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # 2. Run Inference with Recorder
            with recorder:
                # We use generate to get the full reasoning trace
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 3. Process Text
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Remove the prompt from the answer extraction part if needed
            response_only = generated_text[len(prompt):] 
            
            extracted_ans = self.extract_answer(response_only)
            is_correct = self.check_correctness(extracted_ans, ground_truth)
            
            # 4. Process Activations
            # Activations shape: [Total_Tokens, Num_Neurons]
            activations = recorder.get_activations()
            
            # Convert to numpy/list for DataFrame
            # We might have more tokens in activations than generated tokens because it includes the prompt
            # Let's verify lengths or just save all
            act_list = activations.cpu().tolist()
            
            for token_idx, act_values in enumerate(act_list):
                row = {
                    "problem_id": i,
                    "token_index": token_idx,
                    "is_correct": is_correct, # Can be used for "Correct vs Incorrect" analysis
                    "extracted_answer": extracted_ans,
                    "ground_truth": ground_truth
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

    indices_path = 'data.json' # Created by your previous step

    # 2. Run Evaluation
    evaluator = AIMEEvaluator(
        model=model,
        tokenizer=tokenizer,
        neuron_indices_path='data.json',
        layer_name='model.layers.31.mlp.down_proj', # Empty for mock linear, use actual layer for Llama
        use_input=True
    )
    
    evaluator.evaluate(num_samples=2, output_file="aime_results.csv")