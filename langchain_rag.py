from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)


doc1 = Document(page_content="Virat Kohli (Batsman, Former Captain): Scored 8 centuries in 2024, ranking among the top Test and ODI batsmen globally.",
                metadata={"role":"Batsman, Former Captain"})

doc2 = Document(page_content="Rohit Sharma (Batsman, ODI/Test Captain): Led India to the 2024 T20 World Cup victory, ranked third in ICC ODI batting rankings",
                metadata={"role":"Batsman, ODI/Test Captain"})

doc3 = Document(page_content="Jasprit Bumrah (Bowler): Top-ranked Test bowler, leading India’s pace attack with 6/70 in the 2025 England Test series.",
                metadata={"role":"(Bowler)"})

docs=[doc1,doc2,doc3]

vector_store =Chroma(
   embedding_function=OllamaEmbeddings(model="granite-embedding:latest"),
   persist_directory="my_chroma_db",
   collection_name="sample"
)

vector_store.add_documents(docs)

vector_store.get(include=["embeddings","documents","metadatas"])

llm = ChatOllama(model="llama3.2:latest")

prompt_template = """

You are a cricket expert. Answer the question based on the provided context.
Context: {context}

Question: {question}

Answer:

"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

query = "Who among are the top bowlers?"

result = qa_chain.invoke({"query": query})

print("Answer:", result['result'])
print(" \nDocument used :", result['source_documents'][0].page_content)

