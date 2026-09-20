from models.evaluation import MetricResult
def evaluate_correctness(
        response: str,
        expected_facts: list[str],
) -> MetricResult:
    response_lower = response.lower()
    matched = [
        fact
        for fact in expected_facts
        if fact.lower() in response_lower
    ]
    score = (
        len(matched) / len(expected_facts)
        if expected_facts
        else 1.0
    )
    return MetricResult(
        metric="correctness",
        score=score,
        passed=score >= 0.8,
        reason=(
            f"Matched {len(matched)} of "
            f"{len(expected_facts)} expected facts."
        ),
    )