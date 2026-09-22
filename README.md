# LLM Evaluation Framework

A Python-based QA project for evaluating LLM responses using multiple quality metrics.

The framework currently measures:

* correctness
* relevance
* hallucination
* response latency
* overall pass/fail result

## Flow

```text
Test Dataset
    ↓
LLM Response
    ↓
Evaluation
    ├── Correctness
    ├── Relevance
    ├── Hallucination
    └── Latency
    ↓
Overall Score
```

## Project Structure

```text
ai-llm-evaluation-framework/
├── app/
│   └── llm_client.py
├── evaluators/
│   ├── correctness.py
│   ├── relevance.py
│   ├── hallucination.py
│   └── latency.py
├── models/
│   └── evaluation.py
├── datasets/
│   └── qa_cases.json
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```text
OPENAI_API_KEY=your_openai_api_key
USE_MOCK_LLM=false
```

For local testing without API usage:

```text
USE_MOCK_LLM=true
```

## Run

```bash
python main.py
```

The framework evaluates each test case and prints structured results with individual metric scores and an overall pass/fail result.

## Example Metrics

```text
Correctness:    1.0
Relevance:      1.0
Hallucination:  1.0
Latency:        1.0
Overall Score:  1.0
Passed:         true
```

## Current Scope

Implemented:

* JSON-based evaluation dataset
* OpenAI response generation
* mock LLM mode
* correctness evaluation
* relevance evaluation
* hallucination evaluation
* latency measurement
* structured Pydantic models
* overall scoring

## Next Improvements

* richer evaluation datasets
* semantic correctness scoring
* report generation
* configurable thresholds
* pytest integration
* CI execution
* historical result comparison

## Goal

The goal of this project is to explore how LLM behavior can be tested in a structured and repeatable way, rather than validating only whether a response was generated successfully.
