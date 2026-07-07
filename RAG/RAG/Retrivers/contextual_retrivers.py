from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()


model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

compressor = LLMChainExtractor.from_llm(llm)

docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]


vectorstore = FAISS.from_documents(documents=docs, embedding=model)

similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

query = "What is photosynthesis?"


# similarity_results = similarity_retriever.invoke(query)

# contextual_result = ContextualCompressionRetriever.invoke(similarity_results)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=similarity_retriever
)


results = compression_retriever.invoke(query)

for doc in results:
    print(doc.page_content)


# print(contextual_result)




                                #                     User Query
                                #                           │
                                #                           │
                                #                           ▼
                                #         "What is photosynthesis?"
                                #                           │
                                #                           │
                                #                           ▼
                                #          ContextualCompressionRetriever
                                #                  (Main Retriever)
                                #                           │
                                #           ┌───────────────┴────────────────┐
                                #           │                                │
                                #           ▼                                ▼
                                #  Base Retriever (FAISS)          Base Compressor (LLM)
                                #  similarity_retriever            LLMChainExtractor
                                #           │                                ▲
                                #           │                                │
                                #           ▼                                │
                                #  Convert Query to Embedding                │
                                #  (BAAI/bge-base-en-v1.5)                   │
                                #           │                                │
                                #           ▼                                │
                                #  Search FAISS Index                        │
                                #           │                                │
                                #           ▼                                │
                                #  Top-K Documents (k=2)                     │
                                #           │────────────────────────────────┘
                                #           │
                                #           ▼
                                #  Send Retrieved Documents +
                                #  Original Query to LLM
                                #           │
                                #           ▼
                                #  LLM Extracts Only Relevant Parts
                                #           │
                                #           ▼
                                #  Compressed Documents
                                #           │
                                #           ▼
                                #  Returned to User