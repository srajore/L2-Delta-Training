from langchain_ollama import ChatOllama
#from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
#from dotenv import load_dotenv

#load_dotenv(override=True)

# Set up llama3.2:latest with Ollama
llm = ChatOllama(model='llama3.2:latest')

#llm=ChatOpenAI(model='gpt-4o-mini', temperature=0.7, max_tokens=500)

# Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 'You are a friendly assistant named ZenBot.Start with a gretting : Hello. I am ZenBot, your friendly assistant. How can I help you today?'),
    ("human", 'Conversation history :\n {history} \n Current question :{question}'),
])

# Combine the prompt with the LLM
chain = prompt | llm

# Function to get chatbot response
def chatbot(user_input, history_state):
    # Initialize history if empty
    if history_state is None:
        history_state = []

    # Create history string
    history_text = ''

    for question,answer in history_state:
        history_text += "Q : " + question + "\n A: " + answer + "\n"

    response = chain.invoke({
        "question": user_input,
        "history": history_text
        }).content
    

    # Add new question and answer to history
    history_state.append((user_input, response))

    output_text = ""
    for question, answer in history_state:
        output_text += "You : " + question + "\n Assistant: " + answer + "\n"

    return output_text,history_state



def clear_history():
    return "", None  # Reset output and history

# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# ZenBot : Your Friendly Assistant")
    history_state = gr.State(value=None)  # Initialize history state
    input_box = gr.Textbox(label="Your Question", placeholder="Type your question here...")
    output_box = gr.Textbox(label="Answer", interactive=False)
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear History")

    submit_button.click(
        chatbot, 
        inputs=[input_box,history_state], 
        outputs=[output_box,history_state])
    
    clear_button.click(
        clear_history,
        inputs=None,
        outputs=[output_box, history_state]
    )


# Start the Gradio app
demo.launch()












