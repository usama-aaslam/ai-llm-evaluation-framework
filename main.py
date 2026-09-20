import json
from pathlib import Path
from dotenv import load_dotenv

from app.llm_client import get_response
from evaluators.correctness import evaluate_correctness
from evaluators.latency import evaluate_latency
from models.evaluation import (
    EvaluationCase,
    EvaluationResult,
)
DATASET_PATH = Path("datasets/qa_cases.json")
def load_cases() -> list[EvaluationCase]:
    data = json.loads(
        DATASET_PATH.read_text(
            encoding="utf-8",
        )
    )
    return [
        EvaluationCase.model_validate(item)
        for item in data
    ]
def evaluate_case(
        case: EvaluationCase,
) -> EvaluationResult:
    response, latency = get_response(
        case.prompt,
    )
    correctness = evaluate_correctness(
        response,
        case.expected_facts,
    )
    latency_result = evaluate_latency(
        latency,
    )
    metrics = [
        correctness,
        latency_result,
    ]
    overall_score = sum(
        metric.score
        for metric in metrics
    ) / len(metrics)
    return EvaluationResult(
        case_id=case.id,
        prompt=case.prompt,
        response=response,
        metrics=metrics,
        overall_score=overall_score,
        passed=all(
            metric.passed
            for metric in metrics
        ),
    )
def main():
    load_dotenv()
    cases = load_cases()
    for case in cases:
        result = evaluate_case(case)
        print(
            result.model_dump_json(
                indent=2,
            )
        )
if __name__ == "__main__":
    main()