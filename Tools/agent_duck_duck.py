from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
# from langchain import hub
load_dotenv()
search_tool = DuckDuckGoSearchRun()

# result = search_tool.invoke("what is current new of fifa ")

# print(result)


model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
# model.invoke("hi")
# @tool
# def search():
#     """ search the query which is provided """



# prompt = hub.pull("hwchase17/react")

agent = create_agent(
    model = model,
    tools = [search_tool],
    system_prompt="You are a helpful assistant."
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": " ways to go to Antarctica from India"
            }
        ]
    }
)

print(response)