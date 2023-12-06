import os
import datetime
from dotenv import load_dotenv
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from typing import List, Tuple

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_TOKEN')

# Initialize Vector Database
with open('./data/datacontext.txt') as f:
    mysqldbdocs = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=200, 
    length_function=len, 
    is_separator_regex=False)

texts = text_splitter.create_documents([mysqldbdocs])
vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Initialize OpenAI Assistant
assistant = OpenAIAssistantRunnable.create_assistant(
    name="ASkWatson",
    instructions="""
    There is several rules that you need to follow i'm going to rank them from most important to least important.
    If it's less important you are allowed to be more creative and if it's important you need to be more strict. 
    Points are from 1 - 10 with 1 being least important and 10 being most important
    - [10] Your name is ASkWatson, you are an assistant for interactive tasks.
    - [5] If information is available in chat history, use this.
    - [7] Use a friendly and supportive tone in all responses.
    - [6] You are only allowed to answer the question on the following context or Chat History. You may customize the answer to the question.
    - [10] If there is data that needs to be provided by the user, please ask me for this information before answering my question. 
    - [7] You always end the answer with a follow-up question either to ask for extra context or to ask if you can help about a specific topic that is related to the question.
    - [8] Answer the question only on the following context or Chat History.
    If you don't know the answer, just say that you don't know. answer me comprehensively.
    """,
    tools=[{"type": "code_interpreter"}],  # Add other tools if necessary
    model="gpt-3.5-turbo-1106",
)

# Chat history and thread ID
chathistory = []
thread_id = None

def format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "User: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

def process_input(input_text: str, chathistory: List[Tuple], thread_id: str) -> Tuple[str, str]:
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    formatted_input = f"DateTime: {formatted_now} {input_text}"
    chathistory.append((formatted_input, ""))
    chat_history_formatted = format_chat_history(chathistory)

    # Retrieve context from the vector database based on the user's input
    docs = retriever.invoke(input_text)
    
    # If 'docs' are Document objects, access their content directly
    retrieved_context = " ".join([doc.page_content for doc in docs]) if docs else ""

    # Combine the retrieved context with the chat history
    combined_context = "\nContext: " + retrieved_context
    print(combined_context)
    # Prepare input for the assistant with combined context
    assistant_input = {
        "content": formatted_input, 
        "context": combined_context
    }
    if thread_id:
        assistant_input["thread_id"] = thread_id

    # Invoke OpenAI Assistant
    output = assistant.invoke(assistant_input)

    # Extract response and update thread ID
    if isinstance(output, list) and len(output) > 0:
        response = ' '.join([msg.text.value for msg in output[0].content if hasattr(msg.text, 'value')])
        thread_id = output[0].thread_id
    else:
        response = ''
    
    return response, thread_id

print("Initializing chat bot...")
print("Hello, I am ASkWatson, an assistant for interactive tasks. I am here to help you with your questions.")

# Main interaction loop
while True:
    input_text = input("Human: ")
    result, thread_id = process_input(input_text, chathistory, thread_id)
    chathistory[-1] = (chathistory[-1][0], result)
    print("Assistant: " + result)