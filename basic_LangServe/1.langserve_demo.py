import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from langserve import add_routes
import uvicorn


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

parser = StrOutputParser()

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ('user','{text}')
   
])

chain = prompt_template | llm | parser

app = FastAPI(
    title="simpleTranslater",
    description="A simple API server using LangChain's Runnable interfaces",
    version="1.0",
)

add_routes(app,chain,path="/chain")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)