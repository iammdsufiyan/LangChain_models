from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile', temperature= 0.7, max_completion_tokens=1000)

messages = [
    SystemMessage(content="you are a human assistence"),
    HumanMessage(content = "Tell me about langchain")
]


result = model.invoke(messages) # model ke pass bheja input ko invoke means then woh humko palat ke result me jawab diya 

messages.append(AIMessage(content = result.content)) # jo result aya usko humne AIMessage me convert kiya aur phit usko messages me dal diya 

print(messages)