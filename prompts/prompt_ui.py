from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
load_dotenv();
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
      task="text-generation",
      temperature=0.7,
      max_new_tokens=150
)
model = ChatHuggingFace(llm=llm)

st.header('Research Tool')


user_input = st.text_input('Enter your prompt ')


if st.button('Summarize'):
  result = model.invoke(user_input);
  st.write(result.content)

