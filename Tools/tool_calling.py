from langchain_core.tools import  tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


model = ChatGroq(model='llama-3.3-70b-versatile')


def multiply(a:int, b:int) -> int:
    """ multiply the number a and b"""
    return a*b
# tool creation 
@tool
def multiply(a:int, b:int) -> int:
    """ multiply the number a and b"""
    return a*b

# print(multiply.invoke({'a':3, 'b':4}))

# tool binding
llm_with_tools = model.bind_tools([multiply])

# print(llm_with_tools.invoke('Hi how are you'))


query = HumanMessage('can you tell me about yourself')

messages = [query]


# print(messages)


result = llm_with_tools.invoke(messages) # tool calling 

messages.append(result)

# print(result)
print(result.tool_calls[0]['args'])

# tool_result = multiply.invoke(result.tool_calls[0]['args']) is as below line but blow is more good as it is wrapped as tool message and send to llm 

tool_result = multiply.invoke(result.tool_calls[0]) # tool execution step 

messages.append(tool_result)

print(tool_result)

# messages.append(tool_result)


# print(messages)


ans = llm_with_tools.invoke(messages)

print(ans.content)