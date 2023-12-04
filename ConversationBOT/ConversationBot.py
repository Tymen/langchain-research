import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
os.environ['OPENAI_API_KEY'] = "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"

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")

with open('./mysqldb.txt') as f:
    mysqldbdocs = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)

prompt_template = """\
You are an assistant for interactive tasks. Use the following pieces of retrieved context to answer the question with an exception, If the context contains variables like $[variable_name] or $[] anything that uses this pattern the variable name doesn't matter, I want you to ask for these values and explain what information you need from the me, replace the variable with the user's answer. 
If you don't know the answer, just say that you don't know. answer me comprehensively.

The following is a friendly conversation between a human and an AI.
Current conversation:
{history}
Human: {input}

Context: {context}
Answer:
"""

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

PROMPT = PromptTemplate(input_variables=["history", "input", "context"], template=prompt_template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="Assistant"),
)

texts = text_splitter.create_documents([mysqldbdocs])

vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

# prompt = hub.pull("rlm/rag-prompt")

prompt = PromptTemplate.from_template(
    prompt_template
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough() "history": conversation.get_history()}
    | prompt
    | llm
    | StrOutputParser()
)
print(retriever | format_docs)
while True:
    input_text = input("Human: ")
    print(conversation.predict(input=input_text))
print(rag_chain.invoke("How do i fill in the connection string?"))

# cleanup
vectorstore.delete_collection()