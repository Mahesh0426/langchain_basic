import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import  SQLDatabase
from langchain.chains import create_sql_query_chain
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool


load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm = ChatGroq(temperature=0.7, model_name="mistral-saba-24b")

db_path = "db/street_tree_db.sqlite"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# chain = create_sql_query_chain(llm, db=db)
# response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})
# print("\n----------\n")
# print(response)
# print(db.run(response))

execute_query = QuerySQLDataBaseTool(db=db)

write_query = create_sql_query_chain(llm, db)

chain = write_query | execute_query

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)
response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")

print("List the species of trees that are present in San Francisco (passing question and result to the LLM)")

print("\n----------\n")
print(response)

print("\n----------\n")





