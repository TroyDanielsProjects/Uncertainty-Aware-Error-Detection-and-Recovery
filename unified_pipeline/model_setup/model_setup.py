from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict
from tqdm import tqdm
import numpy as np
import json
from dataclasses import asdict

from model_setup.metrics.mech_interp.mech_interp import MechInterp
from model_setup.metric_computer import MetricComputer

logger = logging.getLogger(__name__)

@dataclass(frozen=False)
class Trace:
    text: str
    tokens: List[str]
    explaination: Optional[str] = None
    prompt_length: Optional[int] = None
    mechanistic: Optional[Dict[str, List[float]]] = None
    entropies: Optional[List[float]] = None
    top1_logprobs: Optional[List[float]] = None
    top2_logprobs: Optional[List[float]] = None
    semantic: Optional[List[str]] = None

# Quick fix for numpy types inside a dataclass
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'): # Handles numpy arrays and tensors
            return obj.tolist()
        if hasattr(obj, 'item'):   # Handles numpy scalars (float32)
            return obj.item()
        return super().default(obj)


class ModelSetup:

    mech_interp_recorder = None
    gsm8k_explanation = f"""
    I am going to be asking you to solve a math question.
    Make sure the answer is in this format: ***Answer:[actual answer]***
    Also make sure that only the numbers are present in the answer (no unit, or words)
    So for example, if the answer was 5 inches, it should be formatted like so - ***Answer:5***
    """
    cais_mmlu_explanation = f"""
    I am going to ask you to solve a multiple-choice question. 
    The question is accompanied by a list of choices.

    You must output the final answer as a **single capital letter** (A, B, C, or D) corresponding to the correct choice.
    Make sure the answer is in this format: ***Answer:[Letter]***

    Example: If the answer is "5 inches" and that corresponds to option B, your output should be: ***Answer:B***
    """

    mmlu_pro_explanation = f"""
    I am going to ask you to solve a multiple-choice question.
    The question is accompanied by a list of choices (up to 10 options).

    You must output the final answer as a **single capital letter** (A, B, C, D, E, F, G, H, I, or J) corresponding to the correct choice.
    Make sure the answer is in this format: ***Answer:[Letter]***

    Example: If the answer corresponds to option B, your output should be: ***Answer:B***
    """

    def __init__(
        self,
        model_name: str,
        task: str,
        dtype: torch.dtype,
        uq_methods: list[str],
        mech_interp_ident_methods: list[str],
        entropy_neurons: int,
        data_size: int,
        semantic_runs: int,
        include_prefill: bool,
        device: str,
        explanation: bool,
    ):
        self.device = device
        self.neuron_indices = []
        self.task = task
        self.model_name = model_name
        self.entropic = False
        self.logit_gap = False
        self.semantic = False
        self.heuristic = False
        self.data_size = data_size
        self.semantic_runs = semantic_runs
        self.include_prefill = include_prefill
        self.use_explanation = explanation

        logger.info(f"Initializing ModelSetup for {model_name} on {device}...")
        # will get torch dytpe from string
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device, 
                trust_remote_code=True
            ).eval()

            # Set the tokenizer for the model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Successfully loaded model and tokenizer: {model_name}")

        except Exception as e:
            logger.critical(f"Failed to load model {model_name}. Error: {e}")
            raise e

        if "mech_interp" in uq_methods:
            logger.info("Initializing Mechanistic Interpretability module...")
            try:
                self.mech_interp_recorder, self.neuron_indices = MechInterp.get_recorder(
                    model_name=model_name,
                    model=self.model,
                    mech_interp_ident_methods=mech_interp_ident_methods,
                    entropy_neurons=entropy_neurons
                )
                # get rid of duplicate entropy neurons
                self.neuron_indices = sorted(list(set(self.neuron_indices)))
                logger.info(f"MechInterp attached. Tracking {len(self.neuron_indices)} neurons.")
            except Exception as e:
                logger.error(f"Failed to set up MechInterp: {e}")
                raise e
            
        if "entropic" in uq_methods:
            logger.info("Initializing Entropic Metric")
            self.entropic = True

        if "logit_gap" in uq_methods:
            logger.info("Initializing Logit-Gap Metric")
            self.logit_gap = True
        
        if "semantic" in uq_methods:
            logger.info("Initializing Semantic Metric")
            self.semantic = True

        if "heuristic" in uq_methods:
            logger.info("Initializing heuristic Metric")
            self.heuristic = True
    
    def solve(self, prompt: str, explanation: str = None) -> Trace:
        """
        Generates n_samples solutions.
        Captures each activated metric
        """

        if explanation:
            full_prompt = explanation + prompt
        else: 
            full_prompt = prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        explanation_len = self.token_length(explanation)
        prompt_len = self.token_length(prompt)

        # Helper context manager to handle optional recorder
        # (If recorder is None, we need a dummy context, or just if/else block)
        if self.mech_interp_recorder:
            context_manager = self.mech_interp_recorder
        else:
            # A dummy context manager that does nothing
            from contextlib import nullcontext
            context_manager = nullcontext()
        
        try:
            with context_manager, torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_new_tokens=256,
                    do_sample=False,
                    return_dict_in_generate=True, 
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

        # Process Output
        start = explanation_len if self.include_prefill else explanation_len + prompt_len
        gen_seq = out.sequences[0, start:]
        text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(gen_seq)

        if self.use_explanation:
            trace = Trace(
                explaination=explanation_len,
                prompt_length=len(prompt),
                text=text,
                tokens=tokens
            )
        else:
            trace = Trace(
                prompt_length=len(prompt),
                text=text,
                tokens=tokens
            )

        # Extract Metrics
        if self.entropic or self.logit_gap:

            entropies, t1s, t2s = [], [], []
            for scores in out.scores:
                
                logp = torch.log_softmax(scores[0], dim=-1)
                if self.entropic:
                    p = torch.exp(logp)
                    entropies.append(-torch.sum(torch.nan_to_num(p * logp)).item())
                if self.logit_gap:
                    top2 = torch.topk(logp, 2)
                    t1s.append(top2.values[0].item())
                    t2s.append(top2.values[1].item())
            
            if self.entropic:
                trace.entropies = entropies
                trace.top1_logprobs = t1s
                trace.top2_logprobs = t2s

        # Process Activations
        neuron_activations_dict = None
        if self.mech_interp_recorder:
            neuron_activations_dict = {str(idx): [] for idx in self.neuron_indices}
            activations = self.mech_interp_recorder.get_activations()

            if activations is not None:
                gen_activations = activations[start:, :]
                # Transpose: [Seq_Len, Neurons] -> [Neurons, Seq_Len]
                activations_by_neuron = gen_activations.T.cpu()

                for col_idx, neuron_idx in enumerate(self.neuron_indices):
                    # Ensure key consistency (string vs int)
                    key = str(neuron_idx.item()) if isinstance(neuron_idx, torch.Tensor) else str(neuron_idx)

                    vals = activations_by_neuron[col_idx].tolist()

                    if key not in neuron_activations_dict:
                        neuron_activations_dict[key] = []
                    neuron_activations_dict[key].extend(vals)
                
                trace.mechanistic = neuron_activations_dict

        if self.semantic:
            semantic = []
            for _ in range(self.semantic_runs):
                with torch.no_grad():
                    full_generation = self.model.generate(
                        **inputs, 
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True, 
                        return_dict_in_generate=True, 
                        output_scores=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                    gen_seq = full_generation[0, prompt_len:]
                    text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
                    tokens = self.tokenizer.convert_ids_to_tokens(gen_seq)
                    semantic_entry = {
                        "text": text,
                        "tokens": tokens
                    }
                    semantic.append(semantic_entry)
            trace.semantic = semantic
        return trace
    
    def token_length(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]
        return prompt_len
    
    def run(self, dataset, output_path_results, processed_ids, save_trace = False, output_path_trace = None):
        logger.info(f"Starting run on dataset with {len(dataset)} examples...")
        results = []

        if save_trace and output_path_trace is None:
            logging.info("Save_trace set to true but output_path_trace not given. Defaulting to not saving trace.")
            save_trace = False
        
        if self.data_size == 0:
            self.data_size = len(dataset)

        check_if_exist = processed_ids is not None
        subset = dataset[:self.data_size]

        # A. Generate (Heavy GPU Work)
        explanation = ""
        if self.use_explanation:
            if self.task == 'gsm8k':
                explanation = self.gsm8k_explanation
                logger.info(f"gms8k explanation set")
            elif self.task == "cais_mmlu":
                explanation = self.cais_mmlu_explanation
            elif self.task == "mmlu_pro":
                explanation = self.mmlu_pro_explanation
            else:
                logger.warning(f"Task {self.task} not found when trying to set explaination")

        for i, row in enumerate(tqdm(subset, desc=f"Running Experiment w {self.model_name}")):
            try:
                if check_if_exist:
                    row_id = str(row.get("id", i))
                    # SKIP if already done
                    if row_id in processed_ids:
                        continue
                    
                prompt = f"Question: {row["q"]}\nLet's think step by step."""
                trace = self.solve(prompt, explanation=explanation)
                if not trace: 
                    logger.warning(f"No trace generated for ID {row.get('id', 'unknown')}")
                    continue

                # B. Compute Metrics (CPU Work)
                # We collect all data for this question first
                if self.task == "gsm8k":
                    pred = MetricComputer.extract_answer_gsm8k(trace.text, prompt_to_remove=prompt)
                elif self.task == "cais_mmlu":
                    pred = MetricComputer.extract_answer_mmlu(trace.text, prompt_to_remove=prompt)
                elif self.task == "mmlu_pro":
                    pred = MetricComputer.extract_answer_mmlu_pro(trace.text, prompt_to_remove=prompt)

                is_exact = MetricComputer.check_correctness(pred, row["gold"])

                mechanistic = MetricComputer.aggregate_activations(trace.mechanistic) if trace.mechanistic else None
                avg_entropy = float(np.mean(trace.entropies)) if trace.entropies else None
                gap = MetricComputer.logit_gap(trace.top1_logprobs, trace.top2_logprobs) if trace.top1_logprobs else None
                avg_entropy=float(np.mean(trace.entropies)) if trace.entropies else None
                min_logit_gap=float(np.min(gap)) if gap.size else None
                heuristic_score=MetricComputer.calculate_heuristic_score(trace.text, use_embeddings=True) if self.heuristic else None
                semantic_score = MetricComputer.calculate_semantic_entropy(trace.semantic) if trace.semantic else None
                    
                result_entry = {
                    "id": row.get("id"),
                    "pred": pred,
                    "gold": row["gold"],
                    "is_exact": is_exact,
                    "trace_txt": trace.text,
                    "mechanistic": mechanistic,
                    "entropic": avg_entropy,
                    "min_logit_gap": min_logit_gap,
                    "heurisitc_score": heuristic_score,
                    "semantic_entropy": semantic_score 
                }

                results.append(result_entry)

                with open(output_path_results, "a") as f:
                    # dumps creates the string, write adds it, \n creates the "Line" in JSONL
                    f.write(json.dumps(result_entry, cls=NumpyEncoder) + "\n")

                if save_trace:
                    with open(output_path_trace, "a") as f:
                        f.write(json.dumps(asdict(trace), cls=NumpyEncoder) + "\n")

            except Exception as e:
                logger.error(f"Error processing QID {row.get('id', '?')}: {e}")
                raise e
        return results

