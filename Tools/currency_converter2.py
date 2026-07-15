import json
from typing import Annotated
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool
from langchain_groq import ChatGroq

load_dotenv()

# ==========================================
# 1. DEFINE TOOLS CORRECYLY
# ==========================================

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> dict:
    """Fetches the currency conversion factor between a given base currency and a target currency."""
   
    url = f'https://v6.exchangerate-api.com/v6/ae3da82f382f4b1ddb9f05d3/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Calculates the target currency value from a given base currency value using a conversion rate."""
    return base_currency_value * conversion_rate


# ==========================================
# 2. RUN MANUAL TOOL CALLING PIPELINE
# ==========================================

# Initialize the LLM
llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
# Bind tools (LangChain automatically strips InjectedToolArg from the schema sent to LLM)
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage(content='What is the conversion factor between USD and EUR, and based on that convert 50 USD to EUR')]
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

# Standardize loop parsing to prevent NameErrors
conversion_rate = None
tool_output_map = {}

# First Pass: Execute independent lookup tools
for tool_call in ai_message.tool_calls:
    if tool_call['name'] == 'get_conversion_factor':
        tool_message = get_conversion_factor.invoke(tool_call)
        messages.append(tool_message)
        
        # Safely parse returned string content back to dict
        data = json.loads(tool_message.content) if isinstance(tool_message.content, str) else tool_message.content
        conversion_rate = data.get('conversion_rate')

# Second Pass: Inject runtime parameters into calculation tools
for tool_call in ai_message.tool_calls:
    if tool_call['name'] == 'convert':
        if conversion_rate is None:
            raise ValueError("Cannot convert: conversion_rate was never fetched by the LLM pipeline.")
        
        # Inject the dynamically retrieved rate into arguments
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message = convert.invoke(tool_call)
        messages.append(tool_message)

# Final LLM synthesis
final_response = llm_with_tools.invoke(messages)
print("--- Manual Pipeline Output ---")
print(final_response.content)


# ==========================================
# 3. MODERN AGENT SYSTEM (REPLACING DEPRECATED INITIALIZE_AGENT)
# ==========================================
# Note: For multi-step conditional tools like yours, LangGraph is recommended.
# Here is the correct modern initialization for a quick chat response.

print("\n--- Agent Output ---")
user_query = "Hi how are you?"
agent_response = llm.invoke([HumanMessage(content=user_query)])
print(agent_response.content)
