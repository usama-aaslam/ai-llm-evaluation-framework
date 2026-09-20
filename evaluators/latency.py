from models.evaluation import MetricResult
def evaluate_latency(
        latency_seconds: float,
        max_latency_seconds: float = 5.0,
) -> MetricResult:
    passed = latency_seconds <= max_latency_seconds
    score = min(
        1.0,
        max_latency_seconds / max(
            latency_seconds,
            0.001,
        ),
        )
    return MetricResult(
        metric="latency",
        score=score,
        passed=passed,
        reason=(
            f"Response latency was "
            f"{latency_seconds:.2f}s. "
            f"Threshold: {max_latency_seconds:.2f}s."
        ),
    )