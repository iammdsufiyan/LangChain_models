from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
     task="conversational",
    temperature = 0.7,
    
)

st.header("practice tools ")

model = ChatHuggingFace(llm = llm)

paper_input = st.selectbox("selct a title for artical" ,[" write a paragraph about genai",
   " write a paragraph about machine learning", "write a differenc between genai and machine learning"
])

template = PromptTemplate(
    template="""
    plaese give me point by point about the artical "{paper_input}" with the following specification:
    1-explain mathematical formula
    2-Analogies
    3-why now populer
    """,
    input_variables=['paper_input'],
    validate_template=True
)

if st.button("Generate output"):
    formatted_prompt = template.format(paper_input=paper_input)
    with st.spinner("Thinking..."):
        # Invoke the model and display the result
        response = model.invoke(formatted_prompt)
        st.write(response.content)


