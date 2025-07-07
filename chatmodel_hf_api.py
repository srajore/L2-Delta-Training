from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
#from langchain.chains.conversation.memory  import ConversationBufferMemory
from dotenv import load_dotenv

import os

load_dotenv(override=True)




llm=HuggingFaceEndpoint(
    repo_id='Menlo/Jan-nano-128k',
    task='text-generation', 
    huggingface_api_token=os.getenv('HUGGINGFACE_API_KEY'),  
)

model = ChatHuggingFace(llm=llm)

#memory = ConversationBufferMemory()

#print(memory)

try:
    response = model.invoke("What is the capital of France?")
    print(response.content)
except Exception as e:
    print(f"Error: {e}")
    print("Please check your Hugging Face API key and model configuration.")

