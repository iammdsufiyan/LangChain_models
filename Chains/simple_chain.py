from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()



prompt = PromptTemplate(
    template = 'Generate 5 interesting fact about  {topic}',
    input_variables = ['topic']
)


model = ChatGroq(model='llama-3.3-70b-versatile', temperature= 0.1)

parser = StrOutputParser()


prompt_complete = prompt.invoke({'topic': 'cricket'})


model_output = model.invoke(prompt_complete)


# chain = prompt | model | parser 

# result = chain.invoke({'topic': 'cricket'})

result = parser.invoke(model_output)

print(result)

