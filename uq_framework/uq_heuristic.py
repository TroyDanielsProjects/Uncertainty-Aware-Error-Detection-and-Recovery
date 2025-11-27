
import re
import numpy as np

# ----------------------------------------------------
# Try semantic embeddings; otherwise use simple fallback
# ----------------------------------------------------
_HAS_ST = False
_MODEL = None

try:
    # Attempt import and initialization
    from sentence_transformers import SentenceTransformer, util

    # Load a lightweight model
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    # Uncertain / Certain Anchor Words
    _U_WORDS = [
        "maybe", "perhaps", "unlikely", "doubtful", "unclear", 
        "possible", "guess", "assume", "estimate", "speculate"
    ]
    _C_WORDS = [
        "definitely", "certainly", "proven", "obvious", "undeniable", 
        "fact", "true", "always", "guaranteed", "conclusive"
    ]

    # Encode anchors
    _ANCHORS = _MODEL.encode([" ".join(_U_WORDS), " ".join(_C_WORDS)])
    _U_VEC, _C_VEC = _ANCHORS[0], _ANCHORS[1]

    _HAS_ST = True

except ImportError:
    print("Info: sentence-transformers not found. Using keyword fallback for heuristic UQ.")
except Exception as e:
    print(f"Warning: Error initializing embeddings: {e}. Using fallback.")


# ----------------------------------------------------
# Heuristic Uncertainty Scoring
# ----------------------------------------------------
def analyze_text_confidence(text: str, variant: str = "projection") -> float:
    """
    Analyzes the text for linguistic markers of uncertainty (hedging).
    Higher score indicates higher uncertainty.
    """
    if not text:
        return 0.0

    # ------------------------------------------
    # Fallback (simple lexical hedging detection)
    # ------------------------------------------
    if not _HAS_ST:
        hedges = [
            "might", "perhaps", "possibly", "unclear", "maybe", "assume",
            "unlikely", "probably", "guess", "unsure", "estimate", "approximate",
            "seems", "appears", "could", "suggests"
        ]
        words = text.lower().split()
        if not words: return 0.0
        
        count = sum(1 for w in words if w in hedges)
        
        # Normalize by length. Scaling factor (e.g., 5) adjusted for sensitivity.
        # Score aims to be between 0 and 1.
        return min(1.0, count / (len(words) / 5 + 1))

    # ------------------------------------------
    # Semantic-embedding method
    # ------------------------------------------
    try:
        # Default: global projection
        if variant == "projection":
            full = _MODEL.encode(text)
            fu = util.cos_sim(full, _U_VEC).item() # Similarity to Uncertainty
            fc = util.cos_sim(full, _C_VEC).item() # Similarity to Certainty

            # Net score: (Uncertain - Certain).
            score = fu - fc
            
            # Normalize score (Based on typical ranges for MiniLM-L6-v2, approx -0.25 to 0.25)
            normalized_score = (score + 0.25) / 0.5
            return max(0.0, min(1.0, normalized_score))

        # Other variants can be implemented here
        return 0.0

    except Exception as e:
        # print(f"Error in heuristic calculation (embedding method): {e}")
        return 0.0
