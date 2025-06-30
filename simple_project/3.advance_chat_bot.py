import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)



# response = llm.invoke(mesaageToTheChatbot)
# print(response.content)


chatbotMemory ={}

#input: session_id, output:chatbotMemory[session_id]
def get_session_history (session_id:str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]

chatbot_with_message_history = RunnableWithMessageHistory(
    llm,
    get_session_history
)

session = {"configurable":{"session_id":"123"}}



# First message
# messages = [HumanMessage(content="My favourite language is Python.")]
# response = chatbot_with_message_history.invoke(messages, config=session)
# print("Bot:", response.content)

# Second message, history will be used!
# messages = [HumanMessage(content="What is my favourite language?")]
# response = chatbot_with_message_history.invoke(messages, config=session)
# print("Bot:", response.content)

print("🤖 Chatbot is ready! Type 'exit' to quit.\n")
while True:
    user_input = input(">>: ")
    if user_input.lower() == "exit":
        break
    messages = [HumanMessage(content=user_input)]
    response = chatbot_with_message_history.invoke(messages, config=session)
    print("🤖:", response.content)