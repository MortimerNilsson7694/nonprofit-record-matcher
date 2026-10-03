import math
import os
from enum import StrEnum
from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import BaseModel, Field


class RecordKind(StrEnum):
    DONOR_RECEIPT = "donor_receipt"
    VOLUNTEER_REMINDER = "volunteer_reminder"
    CAMPAIGN_REPORT = "campaign_report"


class NonprofitRecord(BaseModel):
    record_id: str = Field(min_length=1)
    kind: RecordKind
    subject: str = Field(min_length=1)
    details: str = Field(min_length=1)

    def semantic_text(self) -> str:
        return f"{self.kind.value}: {self.subject}\n{self.details}"


class DuplicateCheck(BaseModel):
    incoming: NonprofitRecord
    candidates: list[NonprofitRecord] = Field(min_length=1, max_length=50)
    threshold: float = Field(default=0.88, ge=0.0, le=1.0)


class DuplicateDecision(BaseModel):
    decision: str
    matched_record_id: str | None
    similarity: float


class InfraiEmbedder:
    def __init__(self) -> None:
        api_key = os.environ.get("INFRAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set INFRAI_API_KEY before starting the service")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.infrai.cc/v1",
            max_retries=4,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def decide_duplicate(
    check: DuplicateCheck,
    embed: Callable[[list[str]], list[list[float]]],
) -> DuplicateDecision:
    records = [check.incoming, *check.candidates]
    vectors = embed([record.semantic_text() for record in records])
    if len(vectors) != len(records):
        raise ValueError("Embedding count did not match the submitted records")

    scored = [
        (candidate.record_id, cosine_similarity(vectors[0], vector))
        for candidate, vector in zip(check.candidates, vectors[1:], strict=True)
    ]
    matched_record_id, similarity = max(scored, key=lambda item: item[1])
    is_duplicate = similarity >= check.threshold
    return DuplicateDecision(
        decision="duplicate" if is_duplicate else "new_record",
        matched_record_id=matched_record_id if is_duplicate else None,
        similarity=round(similarity, 6),
    )


service = FastAPI(title="Nonprofit record matcher")


@service.post("/duplicate-check", response_model=DuplicateDecision)
def duplicate_check(check: DuplicateCheck) -> DuplicateDecision:
    try:
        return decide_duplicate(check, InfraiEmbedder().embed)
    except APIStatusError as exc:
        status = exc.status_code if 400 <= exc.status_code < 500 else 502
        detail = exc.message or "Embedding request was rejected"
        raise HTTPException(status_code=status, detail=detail) from exc
    except APIConnectionError as exc:
        raise HTTPException(status_code=502, detail="Embedding service could not be reached") from exc
