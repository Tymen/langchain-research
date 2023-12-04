from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os

os.environ['OPENAI_API_KEY'] = "sk"

llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# now initialize the conversation chain
print (conversation.predict(input="Hi there!"))

