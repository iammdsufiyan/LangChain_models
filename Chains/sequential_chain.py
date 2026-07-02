from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()



prompt1 = PromptTemplate(
    template = "write a report on the {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "generate a  5 point summary from the following summary \n {text}",
    input_variables = ['text']
)

model = ChatGroq(model='llama-3.3-70b-versatile', temperature= 0.1)

parser = StrOutputParser()

chain = prompt1 | model | parser |  prompt2 | model | parser

result = chain.invoke({'topic': 'circket'})


print(result)


print(chain.get_graph().draw_ascii())
