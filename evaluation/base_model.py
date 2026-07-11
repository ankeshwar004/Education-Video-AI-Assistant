from typing import Literal
from pydantic import BaseModel


class BaseQA(BaseModel):
    question: str
    question_type:Literal["factual", "reasoning", "comparison", "application", "misconception"]
    difficulty:Literal["easy", "medium", "hard"]
    answer: str


class FactualQA(BaseQA):
    question_type:Literal["factual"]


class ReasoningQA(BaseQA):
    question_type:Literal["reasoning"]


class ComparisonQA(BaseQA):
    question_type:Literal["comparison"]


class ApplicationQA(BaseQA):
    question_type:Literal["application"]
    scenario:str


class MisconceptionQA(BaseQA):
    question_type:Literal["misconception"]
    misconception:str
    

class FactualQA_List(BaseModel):
    pairs:list[FactualQA]

class ReasoningQA_List(BaseModel):
    pairs:list[ReasoningQA]

class ComparisonQA_List(BaseModel):
    pairs:list[ComparisonQA]
    
class ApplicationQA_List(BaseModel):
    pairs:list[ApplicationQA]
  
class MisconceptionQA_List(BaseModel):
    pairs:list[MisconceptionQA]