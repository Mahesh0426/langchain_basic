import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# chatModel = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
chatModel = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("tell me a curious fact about {soccer_player}")

output_parser = StrOutputParser()

chain = prompt | chatModel | output_parser

# res = chain.invoke({"soccer_player": "Cristiano Ronaldo"})
# print(res)

#batch mode
players = ["Cristiano Ronaldo", "Lionel Messi"]
responses = chain.batch([{"soccer_player": player} for player in players])

for player, fact in zip(players, responses):
    print(f"\n--- {player} ---\n{fact}\n")

# Streaming version 
# print("🤖: ", end="", flush=True)
# for chunk in chain.stream({"soccer_player": "Messi"}):
#     print(chunk, end="", flush=True)
