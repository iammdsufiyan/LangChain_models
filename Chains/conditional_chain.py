from langchain_groq import ChatGroq #     which allows you to use Groq-hosted LLMs inside LangChain.
from langchain_core.output_parsers import StrOutputParser #     parser converts the model output into a plain Python string.
from langchain_core.prompts import PromptTemplate,load_prompt #     PromptTemplate helps create reusable prompts with variables.

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser #     parser forces the model output into a structured Pydantic schema.
from pydantic import BaseModel, Field # * BaseModel is used to define structured data models. Field lets you add metadata like descriptions.
from typing import Literal #     Restricts values to specific allowed choices.
# Replace your old imports with this:
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda 
load_dotenv()
model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser() #     Creates a parser that returns plain text output.

class feedback(BaseModel): # Creating Structured Output Schema
    sentiment : Literal ['positive', 'negative']  =  Field(description =  'Give the sentiment of the feedback')


parser2 = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
        template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
        input_variables=['feedback'],
        partial_variables={'format_instruction':parser2.get_format_instructions()}

)

classifier_chain = prompt1 | model | parser2
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)



chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a worst object , i do not like the shape '}))

print(chain.get_graph().print_ascii())
