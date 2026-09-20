import os
import time

from openai import OpenAI


def get_response(
        prompt: str,
) -> tuple[str, float]:

    if os.getenv("USE_MOCK_LLM") == "true":
        start_time = time.perf_counter()

        mock_response = (
            "We support Visa, Mastercard, and Apple Pay."
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

    start_time = time.perf_counter()

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    latency = (
            time.perf_counter()
            - start_time
    )

    return (
        response.output_text,
        latency,
    )