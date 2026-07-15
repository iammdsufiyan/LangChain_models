from dotenv import load_dotenv
from langchain.vectorestores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# -----------------------------------
# Local Free Embeddings
# -----------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------
# Semantic Chunker
# -----------------------------------
text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season.

The Indian Premier League (IPL) is the biggest cricket league in the world.

Terrorism is a big danger to peace and safety.
"""

docs = text_splitter.create_documents([sample])

print("Number of chunks:", len(docs))



