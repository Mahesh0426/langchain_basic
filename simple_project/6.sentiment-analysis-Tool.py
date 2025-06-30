import os
from dotenv import load_dotenv
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  

load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatGroq(temperature=0.7, model_name="deepseek-r1-distill-llama-70b")

# Define the Prompt for Sentiment Analysis
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passsage.

Only extract the properties that are mentioned in the  'Classification' function.

Passage: {input}
"""
)

# Define the Schema to Extracted Data
class Classification(BaseModel):
    sentiment:str = Field(description="The sentiment of the text")
    political_tendency:str= Field(description="The political tendency of the user")
    language:str = Field(description="The language of the text is written in ")

 # Define the chain to extract the data   
chain = tagging_prompt | llm.with_structured_output(schema=Classification)

trump_followwer = "I love Donald Trump! He is the best president ever. I am so happy he is running for president again. I will vote for him no matter what. He is a great leader and he will make America great again!"

kamala_harris_follower = "Kamala Harris is a great vice president. She is doing a great job and I am so happy she is in office. I will vote for her in the next election. She is a strong leader and she will make America better."

Narendra_Modi_follower = "Narendra Modi is a great prime minister. He is doing a great job and I am so happy he is in office. I will vote for him in the next election. He is a strong leader and he will make India better."
KP_Oli_follower = "K.P. Sharma Oli's leadership has been marred by allegations of corruption and mismanagement, with critics pointing to his administration's failure to address systemic economic issues and accusations of prioritizing political alliances over public welfare."


response = chain.invoke({"input": trump_followwer})

print("Sentiment Analysis of Trump Follower")
print("\n-----------\n")
print(response)

response = chain.invoke({"input": KP_Oli_follower})
print("\n-----------\n")
print("Sentiment Analysis of KP_Oli_follower")
print("\n-----------\n")
print(response)