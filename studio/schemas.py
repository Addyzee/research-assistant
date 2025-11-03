from langgraph.graph import add_messages
from langchain.messages import AnyMessage

from pydantic import BaseModel, Field
from typing import List, Annotated
import operator

class AnalystPersona(BaseModel): #pydantic for validation
    name: str = Field(
        description="Name of the analyst."
    )
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class AnalystPersonas(BaseModel):
    analysts: List[AnalystPersona] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class WholeGraphState(BaseModel):
    # nones are used below for (in)convenience
    max_analysts: int
    topic: str
    max_num_of_sources: int
    human_analyst_feedback: str | None = None
    analysts: list[AnalystPersona] = []
    sections: Annotated[list, operator.add] = []
    introduction: str | None = None
    content: str | None = None 
    conclusion: str | None = None
    final_report: str | None = None
    # document_type: Literal['article', 'research_paper']

class InterviewState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    context: Annotated[list[str], operator.add] = []
    interview: str | None = None
    max_num_of_turns: int = 2
    max_num_of_sources: int
    analyst: AnalystPersona 
    sections: list[str] = []

class InterviewOutputState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    context: Annotated[list[str], operator.add] = []
    interview: str | None = None
    sections: list[str] = []


class SearchQuery(BaseModel):
    search_query: str | None = Field(None, description="Search query for retrieval.")
