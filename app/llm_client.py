import os
import time

from openai import OpenAI


def get_response(
        prompt: str,
        context: str,
) -> tuple[str, float]:

    if os.getenv("USE_MOCK_LLM", "").lower() == "true":
        start_time = time.perf_counter()

        mock_response = get_mock_response(
            prompt,
        )

        latency = (
                time.perf_counter()
                - start_time
        )

        return mock_response, latency

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing."
        )

    client = OpenAI(
        api_key=api_key,
    )

    input_text = f"""
Use the following reference information to answer the question.

Reference:
{context}

Question:
{prompt}

Answer using only information supported by the reference.
"""

    start_time = time.perf_counter()

    response = client.responses.create(
        model="gpt-5-mini",
        input=input_text,
    )

    latency = (
            time.perf_counter()
            - start_time
    )

    return (
        response.output_text,
        latency,
    )


def get_mock_response(
        prompt: str,
) -> str:

    prompt_lower = prompt.lower()

    if "payment" in prompt_lower:
        return (
            "We support Visa, Mastercard, "
            "and Apple Pay."
        )

    if "delivery" in prompt_lower:
        return (
            "Standard delivery takes "
            "2 to 3 business days."
        )

    return (
        "No mock response is configured "
        "for this question."
    )