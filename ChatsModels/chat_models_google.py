from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile', temperature= 1.8, max_completion_tokens=20)

result = model.invoke('tell me a poem in 5 lines ')

print(result.content)
