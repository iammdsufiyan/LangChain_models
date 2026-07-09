from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description = "first number as input ")
    b: int = Field(required=True, description = "second number as input ")

def multiply_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a':3, 'b':3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)