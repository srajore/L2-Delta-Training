from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter


loader = PyPDFLoader("NOTES_Git.pdf") 

docs= loader.load()


splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200, # 10% of overlap
    separator=''
)

result = splitter.split_documents(docs)

print(result[0].page_content)


