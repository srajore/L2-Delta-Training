from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter


loader = PyPDFLoader("NOTES_Git.pdf") 

docs= loader.load()


splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0, # 10% of overlap
    separator=''
)

result = splitter.split_documents(docs)

print(result[0].page_content)


