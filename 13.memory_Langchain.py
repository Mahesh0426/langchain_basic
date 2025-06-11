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
        MessagesPlaceholder(variable_names="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

#buffer memory
