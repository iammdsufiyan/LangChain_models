from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers  import JsonOutputParser
load_dotenv();

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
      task="text-generation",
      temperature=0.7,
      max_new_tokens=150
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate (
    template = 'Give me  name , age , and city of a financial person  \n {format_instruction}' ,
    input_variables = [],
    partial_variables = {'format_instruction' :parser.get_format_instructions() }
)

chain = template1 |  model |  parser

result = chain.invoke({})

print(result)