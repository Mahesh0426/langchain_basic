import os
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# # Load environment variables from .env file
load_dotenv()
# # Initialize the OpenAI chat model with a specified temperature
groq_api_key = os.getenv("GROQ_API_KEY")

groq_llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")

# Define a chat prompt template with placeholders for system, human, and AI messages
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {professional} expert on {topic}."),
    ("human", "Hello, Mr.{professional}, can you please answer a question?"),
    ("ai", "Sure, I can help with that. What is your question?"),
    ("human", "{user_input}"),
])


messages = chat_template.format_messages(
    professional="Gen AI",  
    topic="Generative AI developer",
    user_input="What is the future of  generative AI developer in Australia?"
)

# response = groq_llm.invoke(messages)
# print(response.content)

# stream mode 
for chunk in groq_llm.stream(messages):
    print(chunk.content, end="", flush=True)
