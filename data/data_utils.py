import pandas as pd
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

def get_mock_data(t="reasoning"):
    if t == "reasoning":
        return pd.DataFrame([
            {"question": "If 5 apples cost $10, how much does 1 apple cost?", "answer": "#### 2", "type": "reasoning"},
            {"question": "A train travels 120 miles in 2 hours. What is its speed?", "answer": "#### 60", "type": "reasoning"}
        ])
    return pd.DataFrame([{
        "question": "What is the derivative of sin(x)?\nA. cos(x)\nB. -cos(x)\nC. tan(x)\nD. sec(x)",
        "gold_answer": "A",
        "choices": ["cos(x)", "-cos(x)", "tan(x)", "sec(x)"],
        "type": "multiple_choice"
    }])

def load_gsm8k(split="test", limit=20):
    if load_dataset is None:
        df = get_mock_data("reasoning")
        df["gold_answer"] = df["answer"].apply(lambda x: x.split("####")[-1].strip())
        df["full_solution"] = df["answer"]
        return df[["question", "gold_answer", "full_solution", "type"]]

    try:
        ds = load_dataset("gsm8k", "main", split=split)
    except:
        return load_gsm8k(limit=limit)

    out = []
    for i, x in enumerate(ds):
        if limit and i >= limit:
            break
        gold = x["answer"].split("####")[-1].strip() if "####" in x["answer"] else x["answer"].strip()
        out.append({
            "question": x["question"].strip(),
            "gold_answer": gold,
            "full_solution": x["answer"],
            "type": "reasoning"
        })
    return pd.DataFrame(out)

def load_mmlu(subset="college_mathematics", split="test", limit=20):
    if load_dataset is None:
        return get_mock_data("multiple_choice")

    try:
        ds = load_dataset("cais/mmlu", subset, split=split)
    except:
        return get_mock_data("multiple_choice")

    out = []
    for i, x in enumerate(ds):
        if limit and i >= limit:
            break

        q = x["question"].strip()
        choices = [x["choices"][i] for i in range(len(x["choices"]))]
        gold = x["answer"]

        # Convert to A/B/C/D format when possible
        if isinstance(gold, int) and 0 <= gold < len(choices):
            gold = chr(ord("A") + gold)

        opts = "\n".join(
            f"{chr(ord('A')+j)}. {choices[j]}"
            for j in range(len(choices))
        )

        out.append({
            "question": f"{q}\n{opts}",
            "gold_answer": str(gold),
            "choices": choices,
            "type": "multiple_choice"
        })

    return pd.DataFrame(out)
