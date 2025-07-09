from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv



load_dotenv()

model = ChatOllama(model='llama3.2:latest')

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = 'https://www.amazon.in/Daikin-Inverter-Copper-Filter-MTKM50U/dp/B09R4RYCJ4/ref=sr_1_3?_encoding=UTF8&content-id=amzn1.sym.58c90a12-100b-4a2f-8e15-7c06f1abe2be&dib=eyJ2IjoiMSJ9.D-F4bPyFV7IEMTDZk6Neje-Wx5a-kaJrhcI6JaD_6G0PE2kANyrymdZzhCoNMjavxzsBPbov3DmsVR1XjZGYSZqA0i3DzyclK4FMrFcMdC9OkD5LKQEZfREEwpPLVJnP5kVi7kkQOP6Mh1KxrROHHf48u4asj35i2Z5Q0T8oXGasOH4_cGCi4MaI5au3m1xpLSQowQwhJS9xPZwxn-arNy6s1soNancTCqDSvhzFwqBnu7APuJtW4Yn3bDwSW9xVgJdXIOvX8ENamCQc4-HPKtHmB3OrXkVsR7sloJae47A.sd6Ix9hekdCEXk5ptDJAbw5A_SJRIIREZ0AV1x4jXDg&dib_tag=se&pd_rd_r=1978684e-665b-4fa8-a524-52e0a7a08ce6&pd_rd_w=hEFyQ&pd_rd_wg=B6jVk&qid=1751913866&refinements=p_85%3A10440599031&rps=1&s=kitchen&sr=1-3'
loader = WebBaseLoader(
    url,
    requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
                     "verify": False
                     }
    
)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question': 'What is the product that we are talking about?', 'text': docs[0].page_content}))