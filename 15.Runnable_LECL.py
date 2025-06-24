import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
openai_api_key = os.getenv("GROQ_API_KEY")
# chatModel = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
chatModel = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")

def run(name:str)-> str:
    return f"Hello {name}"

# it does not do any thing but it will return the input value as output
# chain = RunnablePassthrough()
# print(chain.invoke("Bappy"))

#wrap python function as runnable
# chain = RunnablePassthrough() | RunnableLambda(run)
# res = chain.invoke("Bappy")
# print(res)


#it runs task in paralelly
# chain = RunnableParallel(
#     {
#         "task1": RunnablePassthrough(),
#         "task2": RunnableLambda(run),
#     }
# )
# res = chain.invoke("Bappy")
# print(res)

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

def run1(person):
    return person["name"] + " Backham"

chain = RunnableParallel(
    {
        "task1": RunnablePassthrough(),
        "soccer_player": RunnableLambda(run1),
        "task3": RunnablePassthrough()
    }
) | prompt | chatModel | output_parser

res = chain.invoke({
   "name1":"Ronaldo",
   "name":"David"
})
print(res)