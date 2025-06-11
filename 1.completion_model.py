import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

# Load environment variables from .env file
load_dotenv()

openai_api_key=os.getenv("OPENAI_API_KEY")

# completion model 
llm = OpenAI(temperature=0.7)

#simple completion request
# response = llm.invoke("explain me about gen ai Development in 3 sentenses?")
# print(response)``

# streaming response
for chunk in llm.stream("explain me about gen ai Development in 3 sentenses?"):
    print(chunk, end="", flush=True)