from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
VIDEO_ID = "x7X9w_GIm1s"
FILE_PATH = f"transcript_{VIDEO_ID}.txt"

# -----------------------------
# Download transcript
# -----------------------------
if not os.path.exists(FILE_PATH):

    print("Downloading transcript...")

    api = YouTubeTranscriptApi()
    transcript = api.fetch(VIDEO_ID)

    # Save one transcript segment per line
    full_text = "\n".join(
        snippet.text.strip()
        for snippet in transcript
        if snippet.text.strip()
    )

    with open(FILE_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)

    print("Transcript saved!")

else:
    print("Transcript already exists.")

# -----------------------------
# Load transcript
# -----------------------------
loader = TextLoader(FILE_PATH, encoding="utf-8")
documents = loader.load()

# print("Original characters:", len(documents[0].page_content))
# print("Original newlines:", documents[0].page_content.count("\n"))

# ------------------------------------------------
# IMPORTANT:
# Convert every line into a sentence.
# SemanticChunker requires sentence boundaries.
# ------------------------------------------------
lines = []

for line in documents[0].page_content.split("\n"):

    line = line.strip()

    if line:

        lines.append(line)

processed_text = ". ".join(lines) + "."

documents[0].page_content = processed_text

# print("Periods after preprocessing:", processed_text.count("."))

# -----------------------------
# Embedding model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

# -----------------------------
# Semantic Chunker
# -----------------------------
# text_splitter = SemanticChunker(
#     embeddings=embeddings,
#     breakpoint_threshold_type="percentile",
#     breakpoint_threshold_amount=3
# )

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

docs= text_splitter.split_documents(documents)
len(docs)


# docs = text_splitter.split_documents(documents)

# -----------------------------
# Print chunks
# -----------------------------
# print("\nTotal semantic chunks:", len(docs))

# for i, doc in enumerate(docs):
#     print("=" * 80)
#     print(f"Chunk {i+1}")
#     print("=" * 80)
#     print(doc.page_content[:400])
#     print("\nCharacters:", len(doc.page_content))
#     print()

# -----------------------------
# Store in Chroma
# -----------------------------
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="youtube_python_demo"
)

# print("Stored chunks:", len(vectorstore.get()["ids"]))



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# compressor = LLMChainExtractor.from_llm(llm)

# similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})


# query = "What is this about?"

# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=similarity_retriever
# )


# results = compression_retriever.invoke(query)

# print(results)
query = "What happens at the beginning of the video?"

retriever = vectorstore.as_retriever(

    search_type="similarity",

    search_kwargs={"k":8}

)

retrieved_docs = retriever.invoke(query)


print("=" * 80)

for i, doc in enumerate(retrieved_docs):

    print(f"Chunk {i+1}")

    print(doc.page_content)

    print("-" * 80)



context = "\n\n".join(doc.page_content for doc in retrieved_docs)



prompt = f"""
You are an AI assistant answering questions about a YouTube video.

Use ONLY the transcript provided below.

If the transcript does not contain the answer, reply exactly:

"I couldn't find that information in the transcript."

Transcript:
{context}

Question:
{query}

Answer:
"""
response = llm.invoke(prompt)

print(response.content)

