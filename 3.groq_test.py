import os
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# # Load environment variables from .env file
load_dotenv()
# # Initialize the OpenAI chat model with a specified temperature
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")

message = [
    ("system", "You are a helpful AI assistant who is expert in Gen AI ?."),
    ("human", "explain me about gen ai Development in simple sentenses in 3 line?"),
   
]

# simple completion request
# response = llm.invoke(message)
# print(response.content)


# streaming response
for chunk in llm.stream(message):
    print(chunk.content, end="", flush=True)
