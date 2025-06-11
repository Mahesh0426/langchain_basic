import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
# from pydantic import BaseModel
from langchain_openai import ChatOpenAI

load_dotenv()
# Initialize the OpenAI chat model with a specified temperature
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_chatModel = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Define a Pydantic model for the expected output structure
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},  # retun in a JSON format
)

chain = prompt | openai_chatModel | parser

res = chain.invoke({"query": "Tell me a joke on elon musk and his wife tesla"})
print(res)