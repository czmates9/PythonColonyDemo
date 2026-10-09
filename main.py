def analyze_colony(score: float) -> str:
    if score >= 0.9:
        return "high confidence"

    if score >= 0.7:
        return "medium confidence"

    return "low confidence"


scores = [0.95, 0.82, 0.41]

for score in scores:
    result = analyze_colony(score)
    print(f"{score:.2f} -> {result}")