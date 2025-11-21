import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from agents.base_agent import BaseAgent, Trace
from mechanistic_interpretability.activation_utils import ActivationMonitor

class HFAgent(BaseAgent):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", device=None):
        super().__init__(model_name)
        self.model_name = model_name

        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device != "mps" else None,
                trust_remote_code=True
            ).to(self.device)
        except:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)

        self.model.eval()

    def _compute_batch_entropy(self, scores):
        logits = torch.stack(scores)               # [T, B, V]
        logp = torch.log_softmax(logits, -1)       # [T, B, V]
        probs = logp.exp()                         # [T, B, V]

        plogp = probs * logp
        plogp = torch.nan_to_num(plogp, 0.0)
        ent = -torch.sum(plogp, -1)                # [T, B]
        ent = torch.nan_to_num(ent, 0.0)

        topk = torch.topk(logits, 2, -1).values    # [T, B, 2]
        topk_logp = torch.log_softmax(topk, -1)    # [T, B, 2]
        topk_logp = torch.nan_to_num(topk_logp, 0.0)

        return (
            ent.permute(1, 0).cpu().numpy(),       # [B, T]
            topk_logp.permute(1, 0, 2).cpu().numpy()  # [B, T, 2]
        )

    def solve(self, task: str, n_samples: int = 1) -> List[Trace]:
        if "llama" in self.model_name.lower() or "instruct" in self.model_name.lower():
            msgs = [
                {"role": "system", "content": "Solve step-by-step."},
                {"role": "user", "content": task}
            ]
            prompt = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"Question: {task}\nAnswer:"

        prompts = [prompt] * n_samples
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        base_len = inputs.input_ids.shape[1]

        with ActivationMonitor(self.model) as mon:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.6,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

        seq = out.sequences[:, base_len:]
        texts = self.tokenizer.batch_decode(seq, skip_special_tokens=True)

        ent, topk = self._compute_batch_entropy(out.scores)
        acts = mon.get_batch_activations()

        traces = []
        for i in range(n_samples):
            toks = self.tokenizer.convert_ids_to_tokens(seq[i])
            traces.append(Trace(
                text=texts[i],
                tokens=toks,
                entropies=ent[i].tolist(),
                top1_logprobs=topk[i, :, 0].tolist(),
                top2_logprobs=topk[i, :, 1].tolist(),
                activations=acts[i] if len(acts) else None
            ))

        return traces
