import torch
import numpy as np
import json
import os
from transformers import AutoModelForCausalLM
from tqdm import tqdm

def calibrate(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    print(f"Calibrating entropy neurons for {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    layer_idx = len(model.model.layers) - 1
    down = model.model.layers[layer_idx].mlp.down_proj
    head = model.lm_head

    Wn = down.weight.float().detach()          # [hidden, mlp]
    Wv = head.weight.float().detach()          # [vocab, hidden]

    vocab = Wv.shape[0]
    idx = torch.randperm(vocab)[:2048]
    Wv_sub = Wv[idx]

    neuron_mat = Wn
    n = neuron_mat.shape[1]
    bs = 128
    vars_out = []

    for i in tqdm(range(0, n, bs)):
        nb = neuron_mat[:, i:i+bs]
        logits = Wv_sub @ nb                     # [2048, batch]
        vars_out.append(torch.var(logits, 0))

    vars_all = torch.cat(vars_out)

    k = int(n * 0.05)
    top = torch.topk(vars_all, k=k, largest=False).indices

    os.makedirs("config", exist_ok=True)
    path = "config/entropy_neurons_llama3.json"
    with open(path, "w") as f:
        json.dump(top.tolist(), f)

    print(f"Saved {len(top)} indices to {path}")

if __name__ == "__main__":
    calibrate()
