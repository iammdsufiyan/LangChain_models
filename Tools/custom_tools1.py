from langchain_core.tools import tool

def multiply(a,b):
    """" Multiply two number """

    return a*b

 # add type hints
def multiply(a: int ,b: int ) -> int :
    """" Multiply two number """

    return a*b


@tool
def multiply(a: int, b:int) -> int:
    """Multiply two numbers"""
    return a*b

result = multiply.invoke({"a":3, "b":5})

print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema()) # what is actualy passing to llm as json in  this tool calling