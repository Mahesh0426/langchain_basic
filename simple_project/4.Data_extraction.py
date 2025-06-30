import os
from dotenv import load_dotenv
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


#Define the Schema to Extracted Data
class Person(BaseModel):
    """Information about a person"""
    name: Optional[str] = Field(default=None, description="The name of the person")
    lastName: Optional[str] = Field(default=None, description="The lastname of the person if known")
    country: Optional[str] = Field(default=None, description="The country of the person if known")

#Define the Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert extraction algorithm. Only extract relevant information from the text. If you do not know the value of an attribute asked to extract, return null for the attribute's value."),
    ("human", "{text}")
])

chain = prompt | llm.with_structured_output(schema=Person)

comment = "I absolutely love this product! It's been a game-chnager for my daily routine. The quality is top-notch and the customer service is outstanding. I have recommended it to all my friends and family. - Sarah Johnson, USA"
response = chain.invoke({"text": comment})

print(response)