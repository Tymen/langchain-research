import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint
import getpass
import os

os.environ['OPENAI_API_KEY'] = "sk"

# import dotenv

# dotenv.load_dotenv()
# loader = WebBaseLoader(
#     web_paths=("https://siip.app/FAQ.html", "https://siip.group/privacyverklaring/")
#     # bs_kwargs=dict(
#     #     parse_only=bs4.SoupStrainer(
#     #         class_=("post-content", "post-title", "post-header")
#     #     )
#     # ),
# )
# docs = loader.load()

# This is a long document we can split up.
with open('./mysqldb.txt') as f:
    mysqldbdocs = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)

texts = text_splitter.create_documents([mysqldbdocs])

vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

# prompt = hub.pull("rlm/rag-prompt")
prompt_template = """\
You are an assistant for interactive tasks. Use the following pieces of retrieved context to answer the question with an exception, If the context contains variables like $[variable_name] or $[] anything that uses this pattern the variable name doesn't matter, I want you to ask for these values and explain what information you need from the me, replace the variable with the user's answer. 
If you don't know the answer, just say that you don't know. answer me comprehensively.

Question: {question} 

Context: {context} 

Answer:
"""
prompt = PromptTemplate.from_template(
    prompt_template
)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, verbose=True)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(rag_chain.invoke("How do i fill in the connection string?"))

# cleanup
vectorstore.delete_collection()