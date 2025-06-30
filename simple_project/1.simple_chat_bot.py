#LLMChain is deprecated in the newer versions of LangChain. 
# The recommended replacement is to use Runnable interfaces,
# particularly with LCEL (LangChain Expression Language).

 #Deprecated FileChatMessageHistory will also fix in 2 nd version


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory,FileChatMessageHistory



load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

#creating prompt
prompt = ChatPromptTemplate(
    input_variable = ["content","messages"],
    messages= [
        MessagesPlaceholder(variable_name= "chat_history"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

# buffer memory 
memory = ConversationBufferMemory(
    chat_memory= FileChatMessageHistory("chat_history.json"),
    memory_key = "chat_history",
    return_messages=True
)

#llm chain 
llm_chain = LLMChain(
    llm = llm,
    prompt = prompt,
    verbose = False,
    memory = memory
)

#response
# response = llm_chain.invoke({"content": "what is my name "})
# print(response["text"])


# Interactive loop
print("🤖 Chatbot is ready! Type 'exit' to quit.\n")
while True:
    message = input(">> ")
    if message.lower() == "exit":
        for msg in memory.chat_memory.messages:
            print(f"{msg.type.upper()}: {msg.content}")
        break

    print("🤖: ", end="", flush=True)
    for chunk in llm_chain.stream({"content": message}):
        print(chunk["text"], end="", flush=True)
    print()

