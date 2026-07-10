from typing import Literal
from pydantic import BaseModel

class BtwRouteDecision(BaseModel):
    need_web_search:bool

class RouterDecision(BaseModel):
    route:Literal['retrieve', 'verify_claim', 'direct_answer']

class RelevancyDecision(BaseModel):
    is_relevant:bool
    reasoning:str

class SuperSeedingPaper(BaseModel):
    title:str
    url:str
    summary:str
    
class ClainVerificationResult(BaseModel):
    is_superseeded:bool
    verdict_summary:str
    superseeded_papers:list[SuperSeedingPaper]