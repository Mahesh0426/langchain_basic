# The recommended replacement is to use Runnable interfaces,
# particularly with LCEL (LangChain Expression Language).

# in real production  we have to use langgraph for memor
#  instead ot this ConversationBufferMemory

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Setup history
history = FileChatMessageHistory("chat_history.json")

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{content}")
])

# Define memory
memory = ConversationBufferMemory(
    chat_memory=history,
    memory_key="chat_history",
    return_messages=True
)

# Build the LCEL chain (prompt | llm | output parser)
chain = prompt | llm | StrOutputParser()

# Chat loop
print("🤖 Chatbot is ready! Type 'exit' to quit.\n")
while True:
    message = input(">> ")
    if message.lower() == "exit":
        for msg in memory.chat_memory.messages:
            print(f"{msg.type.upper()}: {msg.content}")
        break

    # Invoke the chain with memory
    inputs = {"content": message, "chat_history": memory.chat_memory.messages}
    print("🤖: ", end="", flush=True)
    for chunk in chain.stream(inputs):
        print(chunk, end="", flush=True)

    # Update memory manually (since we're not using LLMChain)
    memory.chat_memory.add_user_message(message)
    memory.chat_memory.add_ai_message(chunk)  # chunk is the last message from the stream
    print()
