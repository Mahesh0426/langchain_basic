# RunnableParallel using default
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
chatModel = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
# chatModel = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    ["Mahesh is a youtuber who provides tutorials on React, Express, Node.js , MongoDB, Postgressql , Python Programming and more in english language"],embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

template = """Answer the qustion based only on the following context:
{context}

QuestionL:{question}
"""

propmt = ChatPromptTemplate.from_template(template)

retriever_chain = (
    RunnableParallel({"context":retriever,"question":RunnablePassthrough()})
    | propmt
    | chatModel
    | StrOutputParser()
)

res = retriever_chain.invoke("who is Mahesh")
print(res)