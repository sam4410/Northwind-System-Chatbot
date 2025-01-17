from pydantic import BaseModel


class NorthwindQueryInput(BaseModel):
    text: str


class NorthwindQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
