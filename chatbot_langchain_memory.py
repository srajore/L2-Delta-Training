from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import gradio as gr
import uuid

# Set up llama3.2:latest with Ollama
llm = ChatOllama(model='llama3.2:latest')

# Create a prompt template with MessagesPlaceholder for history
prompt = ChatPromptTemplate.from_messages([
    ("system", 'You are a friendly assistant named ZenBot. Start with a greeting: Hello. I am ZenBot, your friendly assistant. How can I help you today?'),
    MessagesPlaceholder(variable_name="history"),
    ("human", '{input}')
])

# Initialize chat history store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create a RunnableWithMessageHistory
chain = RunnableWithMessageHistory(
    runnable=prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Function to get chatbot response
def chatbot(user_input, history_state, session_id=str(uuid.uuid4())):
    # Get response from the chain
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    ).content
    
    # Update history for display (Gradio expects a list of [user, assistant] pairs)
    if history_state is None:
        history_state = []
    history_state.append([user_input, response])
    
    return history_state, history_state, session_id

# Function to clear history
def clear_history(session_id):
    if session_id in store:
        store[session_id].clear()  # Clear the conversation history for the session
    return [], None, str(uuid.uuid4())  # Reset output, history, and generate new session ID

# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# ZenBot: Your Friendly Assistant")
    history_state = gr.State(value=None)  # Initialize history state
    session_id = gr.State(value=str(uuid.uuid4()))  # Initialize session ID
    input_box = gr.Textbox(label="Your Question", placeholder="Type your question here...")
    output_box = gr.Chatbot(label="Conversation")
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear History")

    submit_button.click(
        chatbot,
        inputs=[input_box, history_state, session_id],
        outputs=[output_box, history_state, session_id]
    )
    
    clear_button.click(
        clear_history,
        inputs=[session_id],
        outputs=[output_box, history_state, session_id]
    )

# Start the Gradio app
demo.launch()