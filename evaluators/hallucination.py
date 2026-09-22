import os

from agents import Agent, Runner

from models.evaluation import (
    JudgeResult,
    MetricResult,
)


hallucination_agent = Agent(
    name="Hallucination Evaluator",
    instructions="""
You are evaluating whether an AI response contains
unsupported information.

Compare the response only against the provided reference.

Score from 0.0 to 1.0.

1.0 means every factual claim is supported.

0.0 means the response contains major unsupported
or invented claims.

passed must be true when score >= 0.8.

Do not use outside knowledge.
""",
    output_type=JudgeResult,
)


def evaluate_hallucination(
        context: str,
        response: str,
) -> MetricResult:

    if os.getenv(
            "USE_MOCK_LLM",
            "",
    ).lower() == "true":
        return evaluate_mock_hallucination(
            context,
            response,
        )

    judge_input = f"""
Reference:
{context}

AI Response:
{response}
"""

    result = Runner.run_sync(
        hallucination_agent,
        judge_input,
    )

    judge = result.final_output

    return MetricResult(
        metric="hallucination",
        score=judge.score,
        passed=judge.passed,
        reason=judge.reason,
    )


def evaluate_mock_hallucination(
        context: str,
        response: str,
) -> MetricResult:

    context_lower = context.lower()
    response_lower = response.lower()

    known_terms = [
        "visa",
        "mastercard",
        "apple pay",
        "2 to 3 business days",
    ]

    unsupported = [
        term
        for term in known_terms
        if (
                term in response_lower
                and term not in context_lower
        )
    ]

    if unsupported:
        return MetricResult(
            metric="hallucination",
            score=0.5,
            passed=False,
            reason=(
                    "Unsupported information found: "
                    + ", ".join(unsupported)
            ),
        )

    return MetricResult(
        metric="hallucination",
        score=1.0,
        passed=True,
        reason=(
            "No unsupported information "
            "was detected."
        ),
    )