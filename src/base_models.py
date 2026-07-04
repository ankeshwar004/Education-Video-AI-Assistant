from pydantic import BaseModel, Field
from typing import Literal


class Answer(BaseModel):
    response: str=Field(
        description=("""
        A clear educational explanation written in the model's own words.
        Do not quote transcript verbatim unless necessary""")
        )

    timestamps: list[str]=Field(
        default_factory=list,
        description="Relevant timestamp ranges from the video. Empty list if answered from general knowledge only."
    )

    source: Literal["video","general_knowledge","hybrid"]=Field(
        description=(
            "'video'=answered from transcript context. "
            "'general_knowledge'=video didn't cover it, used prior knowledge. "
            "'hybrid'=used both."
        )
    )

    key_takeway: list[str]=Field(
        description="Point out the important point only"
    )

class Summary(BaseModel):
  question: str = Field(description="The reformulated standalone question, NOT an answer")

class Decision(BaseModel):
  answer: bool

class QAPairs(BaseModel):
  question: str
  answer: str

class QAPairsList(BaseModel):
  pairs: list[QAPairs]

class JudgeScore(BaseModel):
    correctness: int
    completeness: int
    faithfulness: int
    clarity: int
    reasoning: str
