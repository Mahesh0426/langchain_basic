import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import  PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# Load a PDF document using PyPDFLoader
loader = PyPDFLoader("data/Lecture2.pdf")
pdf_docs = loader.load()

# print(txt_docs[0].page_content)
# print("LEN",len(txt_docs))

# split into chunks using text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks_of_text = text_splitter.split_documents(pdf_docs)

# print("LEN",len(chunks_of_text))
# print("chunks_of_text ",chunks_of_text[2].page_content)  

#embeddings
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

#store in chroma Db
vector_db = FAISS.from_documents(chunks_of_text,embedding=embedding)

print("store in chroma Db",vector_db)

# retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# invoke
# response = retriever.invoke("explain me about two phase of industry Project Structure ")
# print(response)

#simple use with LCEL - langchain expression language
template = """
You are a helpful assistant. Only answer the question using the context below.

Context:
{context}

Question: {question}

"""


prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model="gpt-4", temperature=0.7)

def format_docs(docs):
    return "\n".join([d.page_content for d in docs])
 
 
#LCEL pipeline
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# response = chain.invoke("give me the summary of this lecture  ? ")
# print(response)

# Streaming version 
print("🤖: ", end="", flush=True)
for chunk in chain.stream("give me the summary of this lecture?"):
    print(chunk, end="", flush=True)
print()  


# run in loop

