from typing import List, Optional

import numpy as np
from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class Field(BaseModel):
    inferText: str
    inferConfidence: str
    bounding: List[Point]


class OcrResponse(BaseModel):
    inferResult: Optional[str] = None
    message: Optional[str] = None
    fields: dict[str, List[Field]]


def convert_ocr_result_to_response_fields(data: dict) -> List[Field]:
    rec_texts: List[str] = data.get("rec_texts", [])
    rec_scores: List[float] = data.get("rec_scores", [])
    rec_polys: List[np.ndarray] = data.get("rec_polys", [])

    fields: List[Field] = []

    for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
        if not isinstance(text, str) or not text.strip():
            continue

        bounding_points = [Point(x=int(x), y=int(y)) for x, y in poly]

        field = Field(
            inferText=text, inferConfidence=f"{score:.3f}", bounding=bounding_points
        )
        fields.append(field)

    return fields
