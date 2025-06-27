import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

import json


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

memory = MemorySaver()

search = TavilySearchResults(max_results=3)

# response = search.invoke("who won the Euro cup 2025?")
# print(response)
# print(json.dumps(response, indent=2))



tools = [search]
agent_executor = create_react_agent(llm,tools,checkpointer=memory)
config = {"configurable":{"thread_id":"001"}}



# response = agent_executor.invoke({"messages":[HumanMessage(content="who won the Euro cup 2025?")]})
# response["messages"]
# # print(response)

# for chunk in agent_executor.stream({"messages":[HumanMessage(content="who won the Euro cup 2025?")]},config):
#     print(chunk, end="", flush=True)
#     print("-------")
# ...existing code...

for chunk in agent_executor.stream({"messages":[HumanMessage(content="what is pyhton ?"
)]}, config):
    if "agent" in chunk:
        for msg in chunk["agent"]["messages"]:
            # Only print AIMessage content
            if msg.__class__.__name__ == "AIMessage":
                print(msg.content)
                print("-------")



