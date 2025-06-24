#ConversationBufferWindowMemory - it is a type of memory in LangChain that keeps
#  track of the last k turns of a conversation.  It drops the oldest messages when 
# there are more than k messages, ensuring that the buffer does not get too large

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
# Load environment variables from .env file
load_dotenv()
# Initialize the OpenAI chat model with a specified temperature
openai_api_key = os.getenv("OPENAI_API_KEY")

# chat model - it takes two input system and human instruction
chatModel = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

#creating prompt
prompt = ChatPromptTemplate(
    messages = [
        SystemMessagePromptTemplate.from_template(
            "you are a nice chatbot having a conversation with a human"
        ),
        # the vairaiable_name here is must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

#buffer memory
window_memory = ConversationBufferWindowMemory( memory_key="chat_history", return_messages=True,k=3)  # it remembers last 3 chats

#conversation chain 
conversation = LLMChain(
    llm=chatModel,
    prompt= prompt,
    memory = window_memory,
    verbose=False
)

# Interactive loop
print("🤖 Chatbot is ready! Type 'exit' to quit.\n")
while True:
    message = input(">> ")
    if message.lower() == "exit":
        for msg in window_memory.chat_memory.messages:
            print(f"{msg.type.upper()}: {msg.content}")
        break

    response = conversation.invoke({"question": message})  
    print("🤖:", response["text"])
    # print(response.keys())

    