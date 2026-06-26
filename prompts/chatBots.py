from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile', temperature= 0.7, max_completion_tokens=1000)

messages = [
    SystemMessage(content="you are a human assistence"),
    
]


while True:
    user_input = input('YOU : ')

    if  user_input == 'exit':
        print("Goodbye")
        break
    messages.append(HumanMessage(content = user_input))
    result = model.invoke(messages)
    print("AI : " ,  result.content)
    messages.append(AIMessage(content = result.content))
   # print(messages)
