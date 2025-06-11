#vector with retriever and faiss db 
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


# Load a TXT document using TextLoader
loader = TextLoader("data/assignment.txt")
txt_docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
chunks_of_text = text_splitter.split_documents(txt_docs)

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vector_db = FAISS.from_documents(chunks_of_text,embedding=embedding)

# print("LEN",len(chunks_of_text))
# print("chunks_of_text ",chunks_of_text[2].page_content)  # 2nd chunk of text

print("store in Faiss Db",vector_db)

retriver = vector_db.as_retriever(search_kwargs={"k": 2})

response = retriver.invoke("how we apply monitoring security ?")
# print(len(response))
print(response[0].page_content)