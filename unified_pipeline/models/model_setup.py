from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict
from tqdm import tqdm
import numpy as np
import json
from dataclasses import asdict

from models.metrics.mech_interp.mech_interp import MechInterp
from models.metric_computer import MetricComputer

logger = logging.getLogger(__name__)

@dataclass(frozen=False)
class Trace:
    text: str
    tokens: List[str]
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

    def __init__(
        self,
        model_name: str,
        task: str,
        dtype: torch.dtype,
        uq_methods: list[str],
        mech_interp_ident_methods: list[str],
        entropy_neurons: int,
        device: str
    ):
        self.device = device
        self.neuron_indices = []
        self.task = task
        self.model_name = model_name
        self.entropic = False
        self.logit_gap = False
        self.semantic = False

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
    
    def solve(self, prompt: str) -> Trace:
        """
        Generates n_samples solutions.
        Captures each activated metric
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

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
        gen_seq = out.sequences[0, prompt_len:] # NOTE - we are getting rid of prompt is this desirable??? could info not be save int the entropy and activations of the prompt???
        text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(gen_seq)

        trace = Trace(
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
                # Remove prompt activations NOTE - may want to change
                gen_activations = activations[prompt_len:, :]
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
            for _ in range(10): # NOTE - change this away from hardcoded
                with torch.no_grad:
                    gen_seq = self.model.generate(
                        **inputs, 
                        max_new_tokens=256,
                        tempature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    ).sequences[0, prompt_len:]
                    text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
                    semantic.append(text)
            trace.semantic = semantic
        return trace
    
    def run(self, dataset, save_trace = False, output_path = None):
        logger.info(f"Starting run on dataset with {len(dataset)} examples...")
        results = []

        if save_trace and output_path is not None:
            logger.info(f"Saving the trace initilized. Clearing possible old file.")
            with open(output_path, "w") as f:
                pass
        else:
            logging.info("Save_trace set to false or output_path not given. Defaulting to not saving trace.")
            save_trace = False

        for row in tqdm(dataset[:1], desc=f"Running Experiment w {self.model_name}"):
            try:
                # A. Generate (Heavy GPU Work)
                prompt = f"Question: {row["q"]}\nLet's think step by step.\nAnswer:"
                trace = self.solve(prompt)
                if not trace: 
                    logger.warning(f"No trace generated for ID {row.get('id', 'unknown')}")
                    continue

                # B. Compute Metrics (CPU Work)
                # We collect all data for this question first
                if self.task == "gsm8k":
                    pred = MetricComputer.extract_answer_gsm8k(trace.text)
                is_exact = MetricComputer.check_correctness(pred, row["gold"])
                mechanistic = MetricComputer.aggregate_activations(trace.mechanistic)
                avg_entropy = float(np.mean(trace.entropies)) if trace.entropies else 0.0
                    
                result_entry = {
                    "id": row.get("id"),
                    "pred": pred,
                    "gold": row["gold"],
                    "trace_txt": trace.text,
                    "is_exact": is_exact,
                    "mechanistic": mechanistic,
                    "entropic": avg_entropy
                }

                results.append(result_entry)

                if save_trace:
                    with open(output_path, "a") as f:
                        json.dump(asdict(trace), f, cls=NumpyEncoder, indent=4)

            except Exception as e:
                logger.error(f"Error processing QID {row.get('id', '?')}: {e}")
                raise e
        return results

