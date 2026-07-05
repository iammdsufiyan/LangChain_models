from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
model = ChatGroq(model='llama-3.3-70b-versatile')

loader = TextLoader('circket.txt', encoding = 'utf-8')


prompt = PromptTemplate(
    template = "write the summary of the text {poem}",
    input_variable=['poem']
)
docs = loader.load()

parser = StrOutputParser()

print(type(docs))

print(len(docs))

print(docs[0])

# print(type(docs[0]))


chain = prompt | model | parser

print(chain.invoke({'poem' : docs[0].page_content}))