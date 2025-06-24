import os
import streamlit as st
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
llm = ChatOpenAI(model="gpt-4", temperature=0.7)


loader = WebBaseLoader(
    web_path=("https://dev.to/alisamir/writing-clean-secure-nodejs-apis-a-checklist-youll-actually-use-3loc?ref=dailydev",),
    bs_kwargs=dict(
        parse_only = bs4.SoupStrainer(
            # class_=("post-content","post-title", "post-header")
        )
    ),
)

docs = loader.load()
# print("loaded docs:",docs)

text_spilitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_spilitter.split_documents(docs)

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vectorDB = Chroma.from_documents(documents=chunks, embedding=embedding)
print("store in chroma Db",vectorDB)

retriever = vectorDB.as_retriever(search_kwargs={"k": 2})

#download inbuilt prompt from hub in Langchain -  https://smith.langchain.com/hub/rlm/rag-prompt
langchain_prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | langchain_prompt
    | llm
    | StrOutputParser()
)



#guardrails
def guarded_rag_chain(question):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    if not context.strip():
        return "I don't know based on the provided information."
    answer = rag_chain.invoke(question)
    return answer

# Usage in your loop:
while True:
    question = input(">> ")
    if question.lower() == "exit":
        break
    print("🤖: ", end="", flush=True)
    answer = guarded_rag_chain(question)
    print(answer)

    
# run in loop
# while True:
#     question = input(">> ")
#     if question.lower() == "exit":
#         break

#     print("🤖: ", end="", flush=True)

#     for chunk in rag_chain.stream(question):
        
#         print(chunk, end="", flush=True)

#     print()  



# res = rag_chain.invoke("what is High Level Components of our RAG System ?")
# print("\n🤖:",res)

# #stream version
# print("🤖: ", end="", flush=True)
# for chunk in rag_chain.stream("what is High Level Components of our RAG System ?"):
#     print(chunk, end="", flush=True)

# while loop 
#  question = input(">> ")
#     if question.lower() == "exit":
#         break
#     response = rag_chain.invoke(question)
#     print("🤖:", response)
