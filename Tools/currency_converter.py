import requests
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage , ToolMessage  
from dotenv import load_dotenv

load_dotenv()
# from langchain_core.tools import tool
API_KEY = "ae3da82f382f4b1ddb9f05d3" 

base_currency = input("Enter base currency: ").strip().upper()

url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{base_currency}"



@tool
def currency(target_currency: str):
    """" this is a currency converter function"""
    response = requests.get(url)
    target = target_currency.strip().upper()

    if response.status_code == 200:
        data = response.json()

       
        output = data["conversion_rates"][target]
        print(f"1 {base_currency} = {output} {target}")
        return f"The exchange rate from {base_currency} to {target} is {output}"
        
    else:
        print(f"Error fetching data: {response.status_code}")
        return None




model = ChatGroq(model='llama-3.3-70b-versatile')


llm_with_tools = model.bind_tools([currency])


query = HumanMessage("what is the currect price of USD")

messages = [query]

result = llm_with_tools.invoke(messages)
messages.append(result)
tool_result = currency.invoke(result.tool_calls[0]) # tool execution step 

tool_message = ToolMessage(content=str(tool_result), tool_call_id=result.tool_calls[0]["id"])
messages.append(tool_message)



final_result = llm_with_tools.invoke(messages)

print("\n--- Final Answer from AI ---")
print(final_result.content)
