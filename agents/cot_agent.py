
import numpy as np
from typing import List
try:
    from openai import OpenAI
    # Assumes OPENAI_API_KEY is set in the environment.
    # Set CLIENT = OpenAI() to enable.
    CLIENT = None 
except ImportError:
    CLIENT = None
    print("Info: 'openai' library not found. CoTAgent requires it for API calls.")
except Exception:
    CLIENT = None

# Ensure correct imports
from agents.base_agent import BaseAgent, Trace

class CoTAgent(BaseAgent):
    """
    Agent utilizing Chain-of-Thought prompting via an external API (e.g., OpenAI).
    """
    def __init__(self, model="gpt-4o-mini"):
        super().__init__(model_name=model)
        self.client = CLIENT
        if self.client is None:
            print(f"Info: OpenAI client not initialized for {model}. CoTAgent will use mock responses.")

    def _entropy(self, logprobs):
        # Calculate entropy from a list of log probabilities
        p = np.exp(logprobs)
        # Normalize probabilities (important if only top-k are provided)
        p_sum = np.sum(p)
        if p_sum > 0:
            p = p / p_sum
        
        # Stable calculation of -sum(p * log(p))
        plogp = np.where(p > 0, p * np.log(p), 0)
        return -np.sum(plogp)

    def _mock(self, task, n):
        # Mock response generator for testing or when API is unavailable
        text = f"Step 1: Analyze '{task[:50]}...'. Step 2: Calculation 10*5=50. Step 3: Review. The final answer is 42."
        toks = text.split()
        # Simulate realistic entropy and logprobs
        e = (np.random.rand(len(toks)) * 1.0 + 0.1).tolist()
        t1 = (-np.random.rand(len(toks)) * 0.2 - 0.05).tolist()
        t2 = [(a - (np.random.rand() * 1.5 + 0.5)) for a in t1]
        
        # Generate mock activations (Shape: T, H). Using a standard MLP dim size (e.g. 14336 for Llama 3 8B)
        D_MLP = 14336
        mock_acts = np.random.randn(len(toks), D_MLP).astype(np.float32) * 0.5 - 0.1
        
        return [Trace(text=text, tokens=toks, entropies=e, top1_logprobs=t1, top2_logprobs=t2, activations=mock_acts)
                for _ in range(n)]

    def solve(self, task: str, n_samples: int = 1) -> List[Trace]:
        if self.client is None:
            return self._mock(task, n_samples)

        prompt = f"Solve the following problem step-by-step.\nQuestion: {task}\nSolution:"

        try:
            # Request generation from the API
            r = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                n=n_samples,
                logprobs=True,  # Crucial for Entropic UQ
                top_logprobs=5, # Get top 5 for accurate entropy calculation
                temperature=0.7 if n_samples > 1 else 0.0 # Use temperature for Semantic UQ
            )
        except Exception as e:
            print(f"API request failed: {e}. Using mock response.")
            return self._mock(task, n_samples)

        out = []
        for c in r.choices:
            lp = getattr(c, "logprobs", None)
            if not lp or not lp.content:
                continue

            # Process the logprobs returned by the API
            toks, ents, t1, t2 = [], [], [], []
            for t in lp.content:
                toks.append(t.token)
                vals = sorted([x.logprob for x in t.top_logprobs], reverse=True)
                
                ents.append(self._entropy(np.array(vals)))
                t1.append(vals[0])
                t2.append(vals[1] if len(vals) > 1 else -np.inf)

            out.append(Trace(
                text=c.message.content.strip(),
                tokens=toks,
                entropies=ents,
                top1_logprobs=t1,
                top2_logprobs=t2,
                activations=None # API agents typically do not return activations
            ))
        return out
