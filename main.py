import json
from pathlib import Path

from dotenv import load_dotenv

from app.llm_client import get_response
from evaluators.correctness import (
    evaluate_correctness,
)
from evaluators.hallucination import (
    evaluate_hallucination,
)
from evaluators.latency import (
    evaluate_latency,
)
from evaluators.relevance import (
    evaluate_relevance,
)
from models.evaluation import (
    EvaluationCase,
    EvaluationResult,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent

DATASET_PATH = (
        PROJECT_ROOT
        / "datasets"
        / "qa_cases.json"
)


def load_cases() -> list[EvaluationCase]:
    data = json.loads(
        DATASET_PATH.read_text(
            encoding="utf-8",
        )
    )

    return [
        EvaluationCase.model_validate(
            item
        )
        for item in data
    ]


def evaluate_case(
        case: EvaluationCase,
) -> EvaluationResult:

    response, latency = get_response(
        case.prompt,
        case.context,
    )

    correctness = evaluate_correctness(
        response,
        case.expected_facts,
    )

    relevance = evaluate_relevance(
        case.prompt,
        response,
    )

    hallucination = (
        evaluate_hallucination(
            case.context,
            response,
        )
    )

    latency_result = evaluate_latency(
        latency,
    )

    metrics = [
        correctness,
        relevance,
        hallucination,
        latency_result,
    ]

    overall_score = sum(
        metric.score
        for metric in metrics
    ) / len(metrics)

    passed = all(
        metric.passed
        for metric in metrics
    )

    return EvaluationResult(
        case_id=case.id,
        prompt=case.prompt,
        response=response,
        metrics=metrics,
        overall_score=round(
            overall_score,
            3,
        ),
        passed=passed,
    )


def main():
    load_dotenv()

    cases = load_cases()

    print(
        f"\nRunning {len(cases)} "
        "LLM evaluation cases...\n"
    )

    for case in cases:
        result = evaluate_case(
            case,
        )

        print(
            result.model_dump_json(
                indent=2,
            )
        )

        print(
            "\n"
            + "-" * 60
            + "\n"
        )


if __name__ == "__main__":
    main()