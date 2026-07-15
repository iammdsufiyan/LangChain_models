from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('check.pdf')

docs = loader.load() # this function help you load the document 

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)