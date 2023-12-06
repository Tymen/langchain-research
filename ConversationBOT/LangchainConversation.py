import os
import datetime
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import format_document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from typing import List, Tuple
max_length = 3400
# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_TOKEN')

# Read and process documents
with open('./data/datacontext.txt') as f:
    mysqldbdocs = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    length_function=len, 
    is_separator_regex=False)

texts = text_splitter.create_documents([mysqldbdocs])
vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Chat history and memory setup
chathistory = []
memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

# Define prompts
from langchain.prompts.prompt import PromptTemplate

# Define prompts as PromptTemplate objects
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
# Given the following conversation and a follow-up question, in the English language.
# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Question:""")

ANSWER_PROMPT = PromptTemplate.from_template("""
There is several rules that you need to follow i'm going to rank them from most important to least important.
If it's less important you are allowed to be more creative and if it's important you need to be more strict. 
Points are from 1 - 10 with 1 being least important and 10 being most important
- [10] Your name is ASkWatson, you are an assistant for interactive tasks.
- [5] If information is available in chat history, use this.
- [10] If there is no response from the Assistent in the chat history you start off introducing yourself. as ASkwatson.
- [7] Use a friendly and supportive tone in all responses.
- [6] You are only allowed to answer the question on the following context or Chat History. You may customize the answer to the question.
- [10] If there is data that needs to be provided by the user, please ask me for this information before answering my question. 
- [7] You always end the answer with a follow-up question either to ask for extra context or to ask if you can help about a specific topic that is related to the question.
- [8] Answer the question only on the following context or Chat History.
                                             
Context:
{context}
This is our past conversation:
Chat History:
{chat_history}

Question: {question}""")


def format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "User's: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

class DocumentWrapper:
    def __init__(self, page_content, context, question, chat_history):
        self.page_content = page_content
        self.metadata = {'context': context, 'question': question, 'chat_history': chat_history}

def process_input(input_text: str, chathistory: List[Tuple]) -> str:
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    formatted_input = f"DateTime: {formatted_now} {input_text}"
    chathistory.append((formatted_input, ""))
    chat_history_formatted = format_chat_history(chathistory)
        # Check if chat history is too long and trim if necessary
    while len(chat_history_formatted) > max_length:  # Define MAX_LENGTH appropriately
        chathistory.pop(0)  # Remove the oldest message
        chat_history_formatted = format_chat_history(chathistory)
    
    # Condense question
    # condensed_question = CONDENSE_QUESTION_PROMPT.format(chat_history=chat_history_formatted, question=formatted_input)
    # standalone_question = ChatOpenAI(temperature=0.5, model="gpt-4").invoke(condensed_question).content
    # print(condensed_question)
    # Retrieve documents
    docs = retriever.invoke(input_text)

    # Wrap documents in the expected format, now including chat_history in metadata
    formatted_docs = [DocumentWrapper(doc['page_content'] if 'page_content' in doc else "", 
                                      doc, 
                                      input_text,
                                      chat_history=chat_history_formatted) # Add chat_history to the metadata
                     for doc in docs]
    
    # Combine documents using the wrapper
    combined_docs = "\n\n".join([format_document(doc, ANSWER_PROMPT) for doc in formatted_docs])
    print(combined_docs)
    # Final answer
    final_input = ANSWER_PROMPT.format(context=combined_docs, question=input_text, chat_history=chat_history_formatted)
    answer = ChatOpenAI(temperature=0.5, model="gpt-4").invoke(final_input).content

    return answer



# Main interaction loop
while True:
    input_text = input("Human: ")
    result = process_input(input_text, chathistory)
    chathistory[-1] = (chathistory[-1][0], result)
    print("Answer: " + result)