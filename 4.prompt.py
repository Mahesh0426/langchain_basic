import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
chatModel = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# example prompt template
prompt = PromptTemplate.from_template("Tell me a {adjective} story about {topic}.")


llm_response = prompt.format(adjective="motivated", topic="The Walton Family")

response = chatModel.invoke(llm_response)
print(response.content)