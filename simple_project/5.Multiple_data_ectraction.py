import os
from dotenv import load_dotenv
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from pydantic import BaseModel, Field  

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

class Person(BaseModel):
    """Information about a person"""
    name: Optional[str] = Field(default=None, description="The name of the person")
    lastName: Optional[str] = Field(default=None, description="The lastname of the person if known")
    country: Optional[str] = Field(default=None, description="The country of the person if known")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert extraction algorithm. Only extract relevant information from the text. If you do not know the value of an attribute asked to extract, return null for the attribute's value."),
    ("human", "{text}")
])

class Data(BaseModel):
    """Extract data about people"""
    people: List[Person]


chain = prompt | llm.with_structured_output(schema=Data)


# custom prompt to provide instruction
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert extraction algorithm. Only extract relevant information from the text. If you do not know the value of an attribute asked to extract, return null for the attribute's value."),
    ("human", "{text}")
])


comment = "Alice Johonson from Canada recently reveived a book she loved. Meanwhile, Bob Smith from USA shared  his insights on the same book in a different review. Both reveiws were very insightful."
response = chain.invoke({"text": comment})

print("\n-----------\n")
print("Key data extractions of a list of entities")
print("\n-----------\n")
print(response)
print("\n-----------\n")


