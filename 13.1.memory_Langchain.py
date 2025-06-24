#ConversationBufferMemory - ConversationBufferMemory is a basic memory implementation in LangChain that stores 
# the entire conversation history in memory without any additional processing.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
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
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#chain
conversation = LLMChain(
    llm=chatModel,
    prompt = prompt,
    verbose = False,
    memory = memory
)
# conversation({"question": "Hello"})
# conversation({"question": "my name is Mahesh and I am learning GEN AI with Langchain"})
# conversation({"question": "Actually I  want to integrate a AI chatbot in my website for customer service support."})
# conversation({"question": "tell me about myself based on previous conversation."})



# Print memory contents
# print(memory.buffer)


# for msg in memory.chat_memory.messages:
#     print(f"{msg.type.upper()}: {msg.content}")

# Interactive loop
print("🤖 Chatbot is ready! Type 'exit' to quit.\n")
while True:
    message = input(">> ")
    if message.lower() == "exit":
        for msg in memory.chat_memory.messages:
            print(f"{msg.type.upper()}: {msg.content}")
        break

    response = conversation.invoke({"question": message})  
    print("🤖:", response["text"])
