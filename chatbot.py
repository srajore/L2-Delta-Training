from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# Set up llama3.2:latest with Ollama
llm = ChatOllama(model='llama3.2:latest')

# Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 'You are a helpful assistant. Answer the questions to the best of your ability.'),
    ("human", '{question}'),
])

# Combine the prompt with the LLM
chain = prompt | llm

# Function to get chatbot response
def chatbot(user_input):
    response = chain.invoke({"question": user_input})
    return response.content

# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Simple Chatbot")
    input_box = gr.Textbox(label="Your Question", placeholder="Type your question here...")
    output_box = gr.Textbox(label="Answer", interactive=False)
    submit_button = gr.Button("Submit")

    submit_button.click(
        chatbot, 
        inputs=input_box, 
        outputs=output_box)


# Start the Gradio app
demo.launch()












