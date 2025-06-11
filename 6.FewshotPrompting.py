import os
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_groq import ChatGroq



# # Load environment variables from .env file
load_dotenv()
# # Initialize the OpenAI chat model with a specified temperature



groq_api_key = os.getenv("GROQ_API_KEY")

groq_llm = ChatGroq(temperature=0.7, model_name="deepseek-r1-distill-llama-70b") #llama3-8b-8192    deepseek-r1-distill-llama-70b

#define the examples
examples = [
    {"input": "how are you ?", "output": "Kasto chha?"},
    {"input": "what is your name ?", "output": "Tapaiko naam ke ho?"},
]

# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a an English to Nepali translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)


chain = final_prompt | groq_llm
res = chain.invoke({"input":"what is the most popular food of nepal?"})
print(res.content)



