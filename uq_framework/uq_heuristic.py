def analyze_text_confidence(text: str) -> float:
    if not text:
        return 0.0

    words = ["might","perhaps","possibly","it seems","unclear","maybe"]
    t = text.lower()

    c = sum(t.count(w) for w in words)
    n = len(text.split())

    return c / n if n else 0.0
