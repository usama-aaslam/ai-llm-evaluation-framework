from pydantic import BaseModel
class EvaluationCase(BaseModel):
    id: str
    prompt: str
    expected_facts: list[str]
class MetricResult(BaseModel):
    metric: str
    score: float
    passed: bool
    reason: str
class EvaluationResult(BaseModel):
    case_id: str
    prompt: str
    response: str
    metrics: list[MetricResult]
    overall_score: float
    passed: bool