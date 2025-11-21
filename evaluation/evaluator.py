import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pointbiserialr
from typing import List, Dict, Any

class Evaluator:
    def __init__(self, results: List[Dict[str, Any]]):
        self.df = self._prepare(results)

    def _prepare(self, results):
        rows = []
        for r in results:
            u = r.get("U_vector")
            if not u: continue
            rows.append({
                "is_correct": r["is_correct"],
                "u_entropy": u["avg_entropy"],
                "u_saup": u["saup_score"],
                "u_semantic": u["semantic_entropy"],
                "u_heuristic": u["heuristic_score"],
                "u_mechanistic": u.get("mechanistic_score", 0.0)
            })
        return pd.DataFrame(rows)

    def evaluate_metric(self, m: str) -> Dict[str, float]:
        if m not in self.df.columns:
            return {"error": f"{m} not found"}

        y_true = self.df["is_correct"]
        y_score = self.df[m]
        y_err = 1 - y_true

        try: auroc = roc_auc_score(y_err, y_score)
        except: auroc = 0.5

        try: corr, p = pointbiserialr(y_true, y_score)
        except: corr, p = 0.0, 1.0

        return {
            "AUROC": round(auroc, 4),
            "Correlation": round(corr, 4),
            "P-Value": round(p, 4)
        }

    def get_summary(self):
        mets = ["u_entropy", "u_saup", "u_semantic", "u_heuristic", "u_mechanistic"]
        out = {}
        for m in mets:
            if m in self.df.columns and self.df[m].std() != 0:
                out[m] = self.evaluate_metric(m)
        return out

if __name__ == "__main__":
    mock = [
        {"is_correct": 1, "U_vector": {"avg_entropy": 0.1, "saup_score": 0.2, "semantic_entropy": 0.0, "heuristic_score": 0.0}},
        {"is_correct": 0, "U_vector": {"avg_entropy": 0.9, "saup_score": 0.8, "semantic_entropy": 0.6, "heuristic_score": 0.1}},
        {"is_correct": 1, "U_vector": {"avg_entropy": 0.2, "saup_score": 0.3, "semantic_entropy": 0.0, "heuristic_score": 0.0}},
    ]
    print(Evaluator(mock).get_summary())
