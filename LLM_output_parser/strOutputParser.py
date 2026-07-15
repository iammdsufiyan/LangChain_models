from dotenv import load_dotenv

from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# -----------------------------------
# Google Free Embeddings
# -----------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# -----------------------------------
# Semantic Chunker
# -----------------------------------
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass.

The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# Create semantic chunks
docs = text_splitter.create_documents([sample])

print("\nTotal Chunks:", len(docs))

# for i, doc in enumerate(docs):
#     print(f"\n========== Chunk {i+1} ==========")
#     print(doc.page_content)

# # -----------------------------------
# # Groq LLM
# # -----------------------------------
# model = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0.1
# )

# response = model.invoke(
#     "Explain semantic chunking in simple words."
# )

# print("\n========== GROQ RESPONSE ==========")
# print(response.content)