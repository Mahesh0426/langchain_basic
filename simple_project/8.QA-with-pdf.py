import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm = ChatGroq(temperature=0.7, model_name="mistral-saba-24b")


file_path = "./assets/db-lecture.pdf"

## Load a TXT document using TextLoader
loader = PyPDFLoader(file_path)
docs = loader.load()

#text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

##split into chunks for processing
chunks_of_text = text_splitter.split_documents(docs)

# create embeddings using OpenAI 
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

#store in chroma Db
vector_db = Chroma.from_documents( chunks_of_text, embedding=embedding)

# Create a retriever from the vector database
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
print("store in chroma Db",vector_db)

# Define the system prompt for the question-answering task
system_prompt = """

You are a helpful assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer as concise as possible.

{context}
 """
# Create a chat prompt template using the system prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the question-answering chain using the prompt and LLM
question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Create the retrieval chain
chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)


# response = chain.invoke({"input": "can you tell me characteristics of Big Data?"})

# print("\n------\n")
# print("Generated Answer:", response["answer"])
# print("\n------\n")

# print("response",response["context"][0].metadata)

while True:
    user_input = input(">>: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting.")
        break
    response = chain.invoke({"input": user_input})
    print("🤖", response["answer"])
  
    