import os

from agents import Agent, Runner

from models.evaluation import (
    JudgeResult,
    MetricResult,
)


relevance_agent = Agent(
    name="Relevance Evaluator",
    instructions="""
You are evaluating whether an AI response is relevant
to the user's question.

Score from 0.0 to 1.0.

1.0 means the response directly and completely
addresses the question.

0.0 means the response is unrelated.

passed must be true when score >= 0.8.

Do not evaluate factual correctness.
Only evaluate relevance.
""",
    output_type=JudgeResult,
)


def evaluate_relevance(
        prompt: str,
        response: str,
) -> MetricResult:

    if os.getenv(
            "USE_MOCK_LLM",
            "",
    ).lower() == "true":
        return evaluate_mock_relevance(
            prompt,
            response,
        )

    judge_input = f"""
Question:
{prompt}

AI Response:
{response}
"""

    result = Runner.run_sync(
        relevance_agent,
        judge_input,
    )

    judge = result.final_output

    return MetricResult(
        metric="relevance",
        score=judge.score,
        passed=judge.passed,
        reason=judge.reason,
    )


def evaluate_mock_relevance(
        prompt: str,
        response: str,
) -> MetricResult:

    prompt_lower = prompt.lower()
    response_lower = response.lower()

    if (
            "payment" in prompt_lower
            and (
            "visa" in response_lower
            or "mastercard" in response_lower
            or "apple pay" in response_lower
    )
    ):
        return MetricResult(
            metric="relevance",
            score=1.0,
            passed=True,
            reason=(
                "The response directly addresses "
                "the payment methods question."
            ),
        )

    if (
            "delivery" in prompt_lower
            and "business days" in response_lower
    ):
        return MetricResult(
            metric="relevance",
            score=1.0,
            passed=True,
            reason=(
                "The response directly addresses "
                "the delivery time question."
            ),
        )

    return MetricResult(
        metric="relevance",
        score=0.0,
        passed=False,
        reason=(
            "The response does not appear "
            "relevant to the question."
        ),
    )