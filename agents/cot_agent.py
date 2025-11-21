import numpy as np
from typing import List
try:
    from openai import OpenAI
    CLIENT = OpenAI()
except Exception:
    CLIENT = None

from agents.base_agent import BaseAgent, Trace

class CoTAgent(BaseAgent):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model)
        self.client = CLIENT

    def _entropy(self, logprobs):
        p = np.exp(logprobs)
        return -np.sum(p * logprobs)

    def _mock(self, task, n):
        text = f"Step 1: Analyze '{task[:30]}...'. Step 2: Compute. Final Answer: 42"
        toks = text.split()
        e = (np.random.rand(len(toks)) * 0.5).tolist()
        t1 = (-np.random.rand(len(toks)) * 0.1).tolist()
        t2 = [(a - (np.random.rand() * 1.5 + 0.5)) for a in t1]
        return [Trace(text=text, tokens=toks, entropies=e, top1_logprobs=t1, top2_logprobs=t2, activations=None)
                for _ in range(n)]

    def solve(self, task: str, n: int = 1) -> List[Trace]:
        if self.client is None:
            return self._mock(task, n)

        prompt = f"Solve step-by-step.\nQuestion: {task}\nSolution:"

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                n=n,
                logprobs=True,
                top_logprobs=5,
                temperature=0.7 if n > 1 else 0
            )
        except Exception:
            return self._mock(task, n)

        out = []
        for c in r.choices:
            lp = getattr(c, "logprobs", None)
            if not lp or not lp.content:
                continue

            toks, ents, t1, t2 = [], [], [], []
            for t in lp.content:
                toks.append(t.token)
                vals = sorted([x.logprob for x in t.top_logprobs], reverse=True)
                ents.append(self._entropy(np.array(vals)))
                t1.append(vals[0])
                t2.append(vals[1] if len(vals) > 1 else 0)

            out.append(Trace(
                text=c.message.content.strip(),
                tokens=toks,
                entropies=ents,
                top1_logprobs=t1,
                top2_logprobs=t2,
                activations=None
            ))
        return out
