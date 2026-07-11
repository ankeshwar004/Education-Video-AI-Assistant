from pydantic import BaseModel
from typing import Literal



class GeneralQA(BaseModel):
  question: str
  question_type: Literal["factual","reasoning","application","comparison"]
  difficulty:Literal["easy","medium","hard"]
  answer: str


class GeneralQA_List(BaseModel):
  pairs: list[GeneralQA]
  
  
class MisconceptionQA(BaseModel):
    misconception: str
    question: str
    difficulty:Literal["easy","medium","hard"]
    answer: str
   

class MisconceptionQA_List(BaseModel):
    qa_pairs: list[MisconceptionQA]
    
