import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
# Initialize the OpenAI chat model with a specified temperature
openai_api_key = os.getenv("OPENAI_API_KEY")

# chat model - it takes two input system and human instruction
chatModel = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

message = [
    ("system", "You are a helpful AI assistant who is expert in Gen AI ?."),
    ("human", "explain me about gen ai Development in simple sentenses in 3 sentenses?"),
   
]

# simple completion request
# response = chatModel.invoke(message)
# print(response.content)   
# print(response.response_metadata)
#print (response.schema)
 #print(response)   - it returns a ChatMessage  all object

# streaming response
for chunk in chatModel.stream(message):
    print(chunk.content, end="", flush=True)
