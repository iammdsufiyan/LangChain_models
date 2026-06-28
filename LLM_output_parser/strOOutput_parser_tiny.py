from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

llm =  HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs = dict(
        temperature = 1.8,
        max_new_tokens = 100
    )

)

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
 template='Write a detailed report on {topic}',
    input_variables=['topic']
)

templet2 = PromptTemplate(

template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})

result1= model.invoke(prompt1)


prompt2 = templet2.invoke({'text':result1.content})

result2 = model.invoke(prompt2)

print(result1.content)